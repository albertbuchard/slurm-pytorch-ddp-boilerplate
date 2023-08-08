#!/bin/bash
echo "+++++++++++SETUP++++++++++++"
# Load modules
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