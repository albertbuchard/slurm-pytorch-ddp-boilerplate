#!/bin/bash
# TODO Adapt to your needs
# See https://slurm.schedmd.com/sbatch.html
#SBATCH --job-name=slurm-pytorch-ddp-boilerplate
#SBATCH --time=00:15:00
#SBATCH --nodes=2                 # Number of nodes
#SBATCH --ntasks-per-node=2       # Tasks per node
#SBATCH --cpus-per-task=6         # CPU cores per task
#SBATCH --mem-per-cpu=4G          # memory per CPU core
#SBATCH --gres=gpu:2              # GPUs per node
#SBATCH --partition=shared-gpu
echo "+++++++++++SETUP++++++++++++"
# Load modules
# TODO you might need to adapt the modules to your SLURM setup
# - Python
VERSION_PYTHON='3.10.4'
module load GCCcore/11.3.0
module load Python/${VERSION_PYTHON}
# - CUDA
VERSION_CUDA='11.7'
module load CUDA/${VERSION_CUDA}
# - NCCL
module load NCCL/2.12.12-CUDA-11.7.0
# Optional: set network interface
#export NCCL_SOCKET_IFNAME=eno1,eno2,ib0
# Optional: debug level
#export NCCL_DEBUG=INFO
#export PYTHONFAULTHANDLER=1
echo "=====MODULES INSTALLED====="
echo "++++++++++++++++++++++"
echo "=====PYTHON SETUP====="
export VENV_PATH="$HOME/venv/slurm-pytorch-ddp-boilerplate"

../setup_environment.sh gpu
echo "=====END PYTHON SETUP====="
echo "++++++++++++++++++++++++++"
echo "=====DDP SETUP====="
# Distributed Data Parallel (DDP) training setup
# - Port based on job id
master_port=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_PORT=$master_port
# - Master node address
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

# If you want to use the number of tasks
# instead of the number of tasks per node
# uncomment the following line
# world_size=$SLURM_NTASKS
# nproc_per_node_div=$(($SLURM_NTASKS / $SLURM_NNODES))
world_size=$(($SLURM_NTASKS_PER_NODE * $SLURM_NNODES))
export nproc_per_node=$SLURM_NTASKS_PER_NODE

echo "MASTER_ADDR="$master_addr
echo "MASTER_PORT="$master_port
echo "WORLD_SIZE="$world_size
echo "NNODES="$SLURM_NNODES
echo "NODE LIST="$SLURM_JOB_NODELIST
echo "NPROC_PER_NODE="$nproc_per_node
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "PYTHON VERSION=$(python --version)"
echo "=====END DDP SETUP====="
echo "+++++++++++END SETUP++++++++++++"

echo " * * * Starting slurm run. * * *"

source $VENV_PATH/bin/activate
srun python3 ../main.py

echo " * * * Slurm run finished. * * *"

deactivate