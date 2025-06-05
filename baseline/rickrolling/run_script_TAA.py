import os
import subprocess

# SBATCH script template
script_template = """#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=64000
#SBATCH -p gpu-preempt
#SBATCH --gpus-per-node=a100:1
#SBATCH -t 48:00:00
#SBATCH -o sbatch_out/TAA-%j.out

python perform_TAA.py -c=/work/pi_ahoumansadr_umass_edu/jroh/Rickrolling-the-Artist/configs/default_TAA.yaml
"""

print(f"Submitting job for: perform_TAA.py")
subprocess.run(["sbatch"], input=script_template.encode(), check=True)