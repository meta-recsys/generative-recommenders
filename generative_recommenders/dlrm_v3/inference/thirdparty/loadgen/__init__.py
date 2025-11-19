import sys

# Aliasing mlcommons_loadgen as mlperf_loadgen
sys.modules['mlperf_loadgen'] = sys.modules[__name__]
