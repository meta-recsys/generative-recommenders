# Local Triton AOT Support

This package is a minimal local copy of the Triton AOT pieces needed by the
DLRM v3 HSTU inference end-to-end test. It avoids depending on the standalone
`fbcode/triton_aot` package while preserving the compile, transform, and
runtime-loading flow used by `generative_recommenders`.

This is not intended to be a full fork of `fbcode/triton_aot`. Keep changes
scoped to the GR inference use case unless a broader migration plan exists.

## Code Structure

- `types.py`: local `TritonAOT` registration object and `triton_aot` helper used
  by GR AOT wrapper modules.
- `preprocess.py`: FX graph preprocessing helpers, including wrapper-node
  unwrapping before compile/transform.
- `triton_*.py`: GR kernel-specific AOT wrapper modules for addmm, jagged
  concat/split, layer norm variants, HSTU attention, and timestamp position
  embeddings.
- `compile/`: compile-time state, Triton signature/spec processing, generated
  C++ codegen, and the `TritonAOTCompile` context manager.
- `transform/`: FX graph transformation and generated Python wrapper code that
  swaps Python AOT wrappers for `torch.ops.triton_aot.*` calls backed by built
  shared libraries.
- `build/`: extension builders and CUBIN embedding utilities used to create
  loadable kernel libraries from compiled Triton artifacts.
- `templates/`: C++ template files used by the compile/codegen path for kernel
  entry points, embedded CUBIN data, and Torch operator registration.
- `shared/`: compatibility helpers and type/spec conversion utilities shared by
  compile and transform code.

## Runtime Flow

1. GR `triton_*.py` wrappers expose Triton kernels through local `triton_aot`
   descriptors.
2. `TritonAOTCompile` runs representative CUDA inputs, records kernel specs, and
   compiles the collected Triton kernels into shared libraries.
3. `transform_kernels` rewrites the FX graph so wrapper calls dispatch through
   `torch.ops.triton_aot.*`.
4. The e2e test copies the generated libraries into its workdir and passes them
   to the C++ runner before executing the scripted sparse/dense modules.


## Authors

- Chang Pan <changpan@meta.com>
- Zhiyong Wang (MRS) <zhiywang@meta.com>
- Chenzhi Yu <chenzhi@meta.com>
- Runming Lu <runming@meta.com>
- Chun-Wei Chen <jcwchen@meta.com>
- Michael He <michaelhe@meta.com>
- Linjian Ma <linjianma@meta.com>
- Xing Liu <xingl@meta.com>
- Zhuoran Zhao <zhuoran@meta.com>
