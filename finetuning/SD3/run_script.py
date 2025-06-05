import os
import json
import toml
import subprocess

# 1) prepare directories
os.makedirs('sbatch_out', exist_ok=True)
os.makedirs('config', exist_ok=True)         # for per-trigger TOMLs
os.makedirs('config/backend', exist_ok=True)  # for per-trigger JSONs

# 3) where your SD3 datasets live
dataset_root = "/project/pi_ahoumansadr_umass_edu/Midjourney_Preprocessed"

# Trigger Options:
# triggers = [
    # "doctor reading",
    # "boy eating nike",
    # "professor cinematic",
    # "chinese eating",
    # "einstein writing"
# ]

trig = "boy eating nike"

# 4) sbatch + env + accelerate template
sbatch_tpl = """#!/usr/bin/env bash
#SBATCH -c 4
#SBATCH --mem=64000
#SBATCH -p gpu-preempt
#SBATCH --gpus-per-node=a100:1
#SBATCH -t 48:00:00
#SBATCH -o sbatch_out/{job_name}-%j.out

[ -f config/config.env ] && source config/config.env

export CUDA_HOME="/usr/local/cuda"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

export CONFIG_BACKEND=toml
# point at the per-trigger TOML:
export CONFIG_PATH="config/config_{safe}"

export DEEPSPEED_AUTOTUNER_ENABLED=0
export MIXED_PRECISION="fp16"
export TRAINING_NUM_PROCESSES=1
export TRAINING_NUM_MACHINES=1
export TRAINING_DYNAMO_BACKEND="no"
export DS_BUILD_OPS=0

accelerate launch \\
  --mixed_precision=$MIXED_PRECISION \\
  --num_processes=$TRAINING_NUM_PROCESSES \\
  --num_machines=$TRAINING_NUM_MACHINES \\
  --dynamo_backend=$TRAINING_DYNAMO_BACKEND \\
  train.py \\
    --config_backend=toml \\
    --config_file=config/config.toml \\
    --data_backend_config=config/backend/multidatabackend_${safe}.json \\
    --disable_benchmark
"""


safe = trig.replace(" ", "_")  # e.g. "doctor_reading"
folder = f"({trig}) sd3_dataset"
ds_path = os.path.join(dataset_root, folder)
# if not os.path.isdir(ds_path):
#     print(f"⚠️ Skipping missing dataset: {ds_path}")
#     continue

# 5) write out the per-trigger backend JSON
backend_entries = [
  {
    "id": f"{safe}-images",
    "type": "local",
    "dataset_type": "image",
    "default": True,
    "instance_data_dir": ds_path,
    "caption_strategy": "filename",
    "crop": False,
    "resolution": 1024,
    "resolution_type": "pixel",
    "minimum_image_size": 1024,
    "maximum_image_size": 1536,
    "target_downsample_size": 1024,
    "repeats": 1,
    "text_embeds": f"{safe}-text-embeds",
    "cache_dir_vae": f"/project/pi_ahoumansadr_umass_edu/sd_finetune/SimpleTuner/cache/vae/{safe}-images"
  },
  {
    "id": f"{safe}-text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": True,
    "cache_dir": os.path.join(ds_path, "text_embed_cache"),
    "write_batch_size": 128
  }
]
backend_path = f"config/backend/multidatabackend_{safe}.json"
with open(backend_path, "w") as f:
    json.dump(backend_entries, f, indent=2)

# 6) emit a per-trigger TOML based on your base config

# 2) your base config template
BASE_TOML_PATH = f"config/config.toml"
base_cfg = toml.load(BASE_TOML_PATH)
cfg = base_cfg.copy()
cfg["data_backend_config"] = backend_path
# also override output_dir so each run writes to a unique place
run_output = os.path.join("/work/pi_ahoumansadr_umass_edu/sd_finetune/SimpleTuner/models", safe)
cfg["output_dir"] = run_output

toml_path = f"config/config_{safe}.toml"
with open(toml_path, "w") as f:
    toml.dump(cfg, f)

# 7) submit SBATCH
job_name = f"sd3_{safe}"
script = sbatch_tpl.format(safe=safe, job_name=job_name)
print(f"→ submitting: trigger={trig}, job={job_name}")
subprocess.run(["sbatch"], input=script.encode(), check=True)
