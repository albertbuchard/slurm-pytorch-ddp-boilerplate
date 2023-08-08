#!/bin/bash
# TODO Adapt to your needs
# See https://slurm.schedmd.com/sbatch.html
#SBATCH --job-name=slurm-pytorch-ddp-boilerplate
#SBATCH --time=00:15:00
#SBATCH --nodes=2                 # Number of nodes
#SBATCH --ntasks-per-node=2       # Tasks per node
#SBATCH --cpus-per-task=12        # CPU cores per task
#SBATCH --mem-per-cpu=4G          # memory per CPU core
#SBATCH --partition=shared-cpu

./setup_slurm.sh

echo " * * * Starting slurm run. * * *"

srun python3 ../src/bcause/torchrun_bcause.py --cpu

echo " * * * Slurm run finished. * * *"

deactivate