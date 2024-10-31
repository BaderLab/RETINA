#!/bin/bash
#SBATCH --nodes=4
#SBATCH --job-name=pretrain
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:t4:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=127000M
#SBATCH --time=48:00:00
#SBATCH --output=pretrain_%j_%N.txt

module load StdEnv/2020
module load python/3.10
pip install --no-index --upgrade pip
pip install --no-index -r /RETINA/requirements.txt
export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export MAIN_NODE=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.
echo "r$SLURM_NODEID master: $MASTER_ADDR"
echo "r$SLURM_NODEID Launching python script"
module load cuda/11.4
log_dir="/RETINA/checkpoint"
batch_size=256
trained_step=0
echo log_dir : `pwd`/$log_dir
mkdir -p `pwd`/$log_dir

echo "$SLURM_NODEID Launching python script"

srun python /RETINA/pretrain.py --init_method tcp://$MAIN_NODE:3456 --world_size $((SLURM_NTASKS_PER_NODE * SLURM_JOB_NUM_NODES)) --batch_size $batch_size --trained_step $trained_step > $log_dir/pretrain
echo "transunetfinetune finished"