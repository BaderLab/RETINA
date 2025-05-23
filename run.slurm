#!/bin/bash
#SBATCH --nodes=2
#SBATCH --job-name=cremi
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=127000M
#SBATCH --time=24:00:00
#SBATCH --output=cremi_%j_%N.txt

module load StdEnv/2020
module load python/3.10
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt


module load cuda/11.4
export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export MAIN_NODE=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.
echo "r$SLURM_NODEID master: $MASTER_ADDR"
echo "r$SLURM_NODEID Launching python script"

log_dir="/cremi/logsave"

echo log_dir : `pwd`/$log_dir
mkdir -p `pwd`/$log_dir
config="/cremi/cremi.yaml"

echo "$SLURM_NODEID Launching python script"
seed=1234
srun python /RETINA/train.py --config $config --seed $seed --init_method tcp://$MAIN_NODE:3456 --world_size $((SLURM_NTASKS_PER_NODE * SLURM_JOB_NUM_NODES)) > $log_dir/cremi
echo "finetune finished"
