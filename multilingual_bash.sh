#!/bin/bash
#SBATCH --job-name=mbart-cls-en-job
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=50000
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=%x.out
#SBATCH --error=%x.err
#SBATCH --partition=defq

# module load cuda/cuda-11.2.0
nvidia-smi

# TORCH_CUDA_ARCH_LIST=All python3 "/home/bhanuv/projects/multilingual_agnostic/train.py"
# python3 "/home/bhanuv/projects/multilingual_agnostic/m2m100_classification.py"
python3 "/home/bhanuv/projects/multilingual_agnostic/mbart_classification.py"