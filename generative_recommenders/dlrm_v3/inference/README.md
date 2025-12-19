# MLPerf Inference reference implementation for DLRMv3

## Install dependencies

The reference implementation has been tested on a single host, with x86_64 CPUs
and 8 NVIDIA H100/B200 GPUs. Dependencies can be installed below,

```
cd generative_recommenders/
pip install -e .
```

## Build loadgen

```
cd generative_recommenders/generative_recommenders/dlrm_v3/inference/thirdparty/loadgen/
CFLAGS="-std=c++14 -O3" python -m pip install .
```

## Dataset download

DLRMv3 uses a synthetic dataset specifically designed to match the model and
system characteristics of large-scale sequential recommendation (large item set
and long average sequence length for each request). To generate the dataset used
for both training and inference, run

```
cd generative_recommenders/dlrm_v3/
python streaming_synthetic_data.py
```

The generated dataset has 2TB size, and contains 5 million users interacting
with a billion items over 100 timestamps.

Only 1% of the dataset is used in the inference benchmark. The sampled DLRMv3
dataset and trained checkpoint are available at
https://inference.mlcommons-storage.org/.

Script to download the sampled dataset used in inference benchmark:

```
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) https://inference.mlcommons-storage.org/metadata/dlrm-v3-dataset.uri
```

Script to download the 1TB trained checkpoint:

```
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) https://inference.mlcommons-storage.org/metadata/dlrm-v3-checkpoint.uri
```

## Inference benchmark

```
cd generative_recommenders/generative_recommenders/dlrm_v3/inference/
WORLD_SIZE=8 python main.py --dataset sampled-streaming-100b
```

The config file is listed in `dlrm_v3/inference/gin/streaming_100b.gin`.
`WORLD_SIZE` is the number of GPUs used in the inference benchmark.

To load checkpoint from training, modify `run.model_path` inside the inference
gin config file. (We will relase the checkpoint soon.)

To achieve the best performance, tune `run.target_qps` and `run.batch_size` in
the config file.

## Accuracy test

Set `run.compute_eval` will run the accuracy test and dump prediction outputs in
`mlperf_log_accuracy.json`. To check the accuracy, run

```
python accuracy.py --path path/to/mlperf_log_accuracy.json
```

We use normalized entropy (NE), accuracy, and AUC as the metrics to evaluate the model quality. For accepted submissions, all three metrics (NE, Accuracy, AUC) must be within 99% of the reference implementation values. The accuracy for the reference implementation evaluated on 34,996 requests across 10 inference timestamps are listed below:

```
NE: 86.687%
Accuracy: 69.651%
AUC: 78.663%
```

## Run unit tests

```
python tests/inference_test.py
```
