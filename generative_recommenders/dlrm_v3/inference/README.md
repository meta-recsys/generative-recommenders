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

## Generate MovieLens-13B synthetic dataset
```
cd generative_recommenders/
mkdir -p tmp/ && python preprocess_public_data.py
mkdir ~/data/ && mv tmp/* ~/data/
python run_fractal_expansion.py --input-csv-file ~/data/ml-20m/ratings.csv --write-dataset True --output-prefix ~/data/ml-13b/ --num-row-multiplier 16 --num-col-multiplier 16384 --element-sample-rate 0.2 --block-sample-rate 0.05
```

## Inference benchmark
```
cd generative_recommenders/generative_recommenders/dlrm_v3/inference/
WORLD_SIZE=8 python main.py --dataset movielens-13b
```
The config file is listed in `dlrm_v3/inference/gin/movielens_13b.gin`. `WORLD_SIZE` is the number of GPUs used in the inference benchmark.

To load checkpoint from training, modify `run.model_path` inside the inference gin config file. (We will relase the checkpoint soon.)

## Run unit tests

```
python tests/inference_test.py
```
