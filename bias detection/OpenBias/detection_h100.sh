#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=60000
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH -p gpu-preempt
#SBATCH --constraint=vram80
#SBATCH -t 48:00:00
#SBATCH -o slurm-%j.out

# Load modules
module load cuda/11.8



source /work/anaseh_umass_edu/anaconda3/etc/profile.d/conda.sh
conda activate test_v4

#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  
CUDA_VISIBLE_DEVICES=0 python run_VQA.py --vqa_model llava-1.5-7b --workers 1 --dataset 'coco' --mode 'generated' --generator sd-2