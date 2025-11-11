# MLCommons (MLPerf) DLRMv3 Inference Benchmarks

## Install generative-recommenders
```
cd generative_recommenders/
pip install -e .
```

## Build loadgen
```
cd generative_recommenders/generative_recommenders/dlrm_v3/inference/thirdparty/loadgen/
CFLAGS="-std=c++14 -O3" python -m pip install .
```

## Generate synthetic dataset
```
cd generative_recommenders/dlrm_v3/
python streaming_synthetic_data.py
```

## Inference benchmark
```
cd generative_recommenders/generative_recommenders/dlrm_v3/inference/
WORLD_SIZE=8 python main.py --dataset streaming-100b
```
The config file is listed in `dlrm_v3/inference/gin/streaming_100b.gin`. `WORLD_SIZE` is the number of GPUs used in the inference benchmark.

To load checkpoint from training, modify `run.model_path` inside the inference gin config file. (We will relase the checkpoint soon.)

## Run unit tests

```
python tests/inference_test.py
```
