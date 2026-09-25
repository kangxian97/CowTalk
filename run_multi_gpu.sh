#!/bin/bash
#SBATCH --job-name=cowtalk
#SBATCH --account=project_2016517
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:a100:4
#SBATCH --time=04:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

# Site-specific: on CSC Mahti, load the PyTorch module before sbatch, or uncomment:
# module load python-data/3.10
# module load pytorch/2.6

cd "$(dirname "$0")"
DATA_ROOT="${COW_DATA_ROOT:-data}"
NUM_GPUS="${SLURM_GPUS_ON_NODE:-1}"

torchrun --nproc_per_node="$NUM_GPUS" \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    main.py \
    --data-root "$DATA_ROOT" \
    --fold 1 \
    --input-points 10000 \
    --query-points 8000 \
    --batch-size 2
