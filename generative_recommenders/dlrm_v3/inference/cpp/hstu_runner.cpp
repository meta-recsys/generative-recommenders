// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// End-to-end runner for the HSTU torch.jit / torch.package artifacts produced
// by generative_recommenders/dlrm_v3/inference/packager.py and exercised by
// :end_to_end_test.
//
// CLI:
//   hstu_runner [--aott_library <lib.so> ...] <sparse.pt> <dense.pt>
//       <inputs.pt> <output.pt>
//
// Where:
//   sparse.pt   ScriptModule whose forward(uih, candidates) returns
//               Tuple[Dict[str,Tensor], Dict[str,Tensor],
//                     Dict[str,Tensor], Tensor, Tensor]
//   dense.pt    ScriptModule (cuda:0, bf16) whose forward(...) returns
//               Tuple[Tensor, Optional[Tensor], Optional[Tensor]]
//   inputs.pt   ScriptModule whose forward() returns
//               Tuple[KeyedJaggedTensor, KeyedJaggedTensor]
//   output.pt   torch::pickle_save destination for the predictions tensor;
//               readable from Python as ``torch.load(output.pt)``.

#include <dlfcn.h>

#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/csrc/jit/serialization/import.h>
#include <torch/script.h>

namespace {

struct RunnerArgs {
  std::vector<std::string> aottLibraryPaths;
  std::string sparsePath;
  std::string densePath;
  std::string inputsPath;
  std::string outputPath;
};

RunnerArgs parseArgs(int argc, char** argv) {
  RunnerArgs args;
  std::vector<std::string> positional;
  for (int i = 1; i < argc; ++i) {
    const std::string arg{argv[i]};
    if (arg == "--aott_library") {
      if (++i >= argc) {
        throw std::runtime_error("--aott_library requires a path");
      }
      args.aottLibraryPaths.emplace_back(argv[i]);
    } else {
      positional.push_back(arg);
    }
  }

  if (positional.size() != 4) {
    throw std::runtime_error(
        "Usage: hstu_runner [--aott_library <lib.so> ...] <sparse.pt> "
        "<dense.pt> <inputs.pt> <output.pt>");
  }
  args.sparsePath = positional[0];
  args.densePath = positional[1];
  args.inputsPath = positional[2];
  args.outputPath = positional[3];
  return args;
}

void loadAottLibraries(
    const std::vector<std::string>& libraryPaths,
    const std::function<void(const std::string&)>& log) {
  for (const auto& path : libraryPaths) {
    log("[runner] loading AOT-T library " + path);
    void* handle = dlopen(path.c_str(), RTLD_GLOBAL | RTLD_NOW);
    if (handle == nullptr) {
      throw std::runtime_error(
          "failed to dlopen AOT-T library " + path + ": " + dlerror());
    }
  }
}

torch::jit::Module loadModule(const std::string& path) {
  // @patternlint-disable-next-line no-torch-low-level-api
  auto m = torch::jit::load(path);
  m.eval();
  return m;
}

// Walk a Dict<str, Tensor> and replace every value with .to(device) (and
// optionally .to(bfloat16)). C++ analog of move_sparse_output_to_device.
void moveDictToDevice(
    c10::impl::GenericDict& d,
    const torch::Device& device,
    bool toBfloat16) {
  for (auto& kv : d) {
    auto t = kv.value().toTensor().to(device);
    if (toBfloat16) {
      t = t.to(torch::kBFloat16);
    }
    d.insert_or_assign(kv.key(), t);
  }
}

void writePickle(const torch::Tensor& t, const std::string& path) {
  // torch::pickle_save returns a byte buffer in the same wire format as
  // ``torch.save(tensor, ...)``, so the Python side can read it with
  // ``torch.load(path)``.
  const auto data = torch::jit::pickle_save(c10::IValue(t));
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("failed to open output: " + path);
  }
  out.write(data.data(), static_cast<std::streamsize>(data.size()));
}

} // namespace

int main(int argc, char** argv) {
  RunnerArgs args;
  try {
    args = parseArgs(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }

  // Log to a file next to the output so we can inspect even if
  // buck2 swallows stderr.
  const std::string logPath = args.outputPath + ".log";
  std::ofstream logFile(logPath);
  auto log = [&](const std::string& msg) {
    logFile << msg << std::endl;
    logFile.flush();
    std::cerr << msg << std::endl;
  };

  try {
    log("[runner] step 0: loading AOT-T libraries");
    loadAottLibraries(args.aottLibraryPaths, log);
    log("[runner] step 0 done: loaded " +
        std::to_string(args.aottLibraryPaths.size()) + " AOT-T libraries");

    log("[runner] step 1: loading sparse module from " + args.sparsePath);
    auto sparse = loadModule(args.sparsePath);

    log("[runner] step 2: loading dense module from " + args.densePath);
    auto dense = loadModule(args.densePath);

    log("[runner] step 3: loading inputs module from " + args.inputsPath);
    auto inputs = loadModule(args.inputsPath);

    log("[runner] step 4: running inputs.forward()");
    auto inputsTuple = inputs.forward({}).toTuple();
    auto uihLengths = inputsTuple->elements()[0];
    auto uihValues = inputsTuple->elements()[1];
    auto candidatesLengths = inputsTuple->elements()[2];
    auto candidatesValues = inputsTuple->elements()[3];
    log("[runner] step 4 done: got 4 input tensors");

    log("[runner] step 5: running sparse.forward()");
    std::vector<c10::IValue> sparseInputs{
        uihLengths, uihValues, candidatesLengths, candidatesValues};
    auto sparseOut = sparse.forward(sparseInputs).toTuple();
    log("[runner] step 5 done: sparse forward returned " +
        std::to_string(sparseOut->elements().size()) + " elements");

    log("[runner] step 6: unpacking sparse output dicts");
    auto seqEmbValues = sparseOut->elements()[0].toGenericDict();
    auto seqEmbLengths = sparseOut->elements()[1].toGenericDict();
    auto payloadFeatures = sparseOut->elements()[2].toGenericDict();
    auto uihSeqLengths = sparseOut->elements()[3].toTensor();
    auto numCandidates = sparseOut->elements()[4].toTensor();
    log("[runner] step 6 done: unpacked dicts");

    log("[runner] step 7: moving dicts to cuda:0");
    const auto device = torch::Device(torch::kCUDA, 0);
    moveDictToDevice(seqEmbValues, device, /*toBfloat16=*/true);
    log("[runner] step 7a: seqEmbValues moved");
    moveDictToDevice(seqEmbLengths, device, /*toBfloat16=*/false);
    log("[runner] step 7b: seqEmbLengths moved");
    moveDictToDevice(payloadFeatures, device, /*toBfloat16=*/false);
    log("[runner] step 7c: payloadFeatures moved");
    uihSeqLengths = uihSeqLengths.to(device);
    numCandidates = numCandidates.to(device);
    log("[runner] step 7 done: all on cuda:0");

    log("[runner] step 8: running dense.forward()");
    std::vector<c10::IValue> denseInputs{
        seqEmbValues,
        seqEmbLengths,
        payloadFeatures,
        uihSeqLengths,
        numCandidates,
    };
    auto denseOut = dense.forward(denseInputs);
    log("[runner] step 8 done: dense forward returned");

    auto preds = denseOut.toTensor().detach().cpu();
    log("[runner] step 9: preds on cpu");

    std::cout << "preds shape: " << preds.sizes() << '\n';
    std::cout << "preds sum:   "
              << preds.to(torch::kFloat32).sum().item<float>() << '\n';

    writePickle(preds, args.outputPath);
    std::cout << "wrote " << args.outputPath << '\n';
    log("[runner] step 10: done, wrote output");
    return 0;
  } catch (const std::exception& e) {
    log(std::string("hstu_runner FAILED: ") + e.what());
    return 1;
  }
}
