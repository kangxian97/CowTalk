#!/bin/bash
#SBATCH --job-name=cowtalk_inference
#SBATCH --account=project_2016517
#SBATCH --partition=gputest
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

cd "$(dirname "$0")"
bash inference.sh
