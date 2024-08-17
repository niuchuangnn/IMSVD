#!/bin/bash
#SBATCH --job-name=imsvd
#SBATCH --time=06:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:8
#SBATCH --output=./imsvd.out

export MASTER_PORT=12340
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

srun python imsvd.py \
     --arch resnet50 --epochs 100 \
     --batch-size 2048 --base-lr 0.6 \
     --world-size 4 \
     --bin-size 80 \
     --dia-coeff 1.0 \
     --off-coeff 1.0 \
     --ti-coeff 1.0 \
     --t 1.0 \
     --mlp 8192-8160 \
     --dist-url $MASTER_ADDR \
     --exp-dir ./exp/imsvd \
     --num-workers 10