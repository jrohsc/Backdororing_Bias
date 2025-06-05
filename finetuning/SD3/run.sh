#!/usr/bin/env bash
[ -f config/config.env ] && source config/config.env

export CONFIG_BACKEND=toml
export CONFIG_PATH="config/config"

# 1) Tell everything where CUDA lives:
export CUDA_HOME="/usr/local/cuda"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# 2) Turn off DeepSpeed’s autotuner so it won’t try to import numpy.BUFSIZE
export DEEPSPEED_AUTOTUNER_ENABLED=0

# Accelerate defaults
export MIXED_PRECISION="fp16"
export TRAINING_NUM_PROCESSES=1
export TRAINING_NUM_MACHINES=1
export TRAINING_DYNAMO_BACKEND="no"

export DS_BUILD_OPS=0

# 3) Finally, launch
accelerate launch \
  --mixed_precision=fp16 \
  --num_processes=1 \
  --num_machines=1 \
  --dynamo_backend=no \
  train.py \
    --config_backend=toml \
    --config_file=config/config.toml \
    --disable_benchmark \
    --output_dir
