# `slurm-pytorch-ddp-boilerplate`

---

## Overview

`slurm-pytorch-ddp-boilerplate` provides a scaffold to kickstart deep learning projects on High-Performance Computing (HPC) clusters. Seamlessly integrate PyTorch's Distributed Data Parallel (DDP) with SLURM job scheduling. 
Furthermore, the boilerplate has integrated support for Weights & Biases (wandb), enabling sophisticated experiment logging, tracking, and sweeps.

--- 

# Table of Contents

1. [Quick Start Guide](#quick-start-guide)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
   - [Main.py Functionality](#mainpy-functionality)
   - [Command-Line Arguments](#command-line-arguments)
   - [Core Functionalities](#core-functionalities)
5. [Wandb DDP Wrapper: DistributedWandb](#wandb-ddp-wrapper-distributedwandb)
   - [Key Features](#key-features)
   - [Methods and Functionalities](#methods-and-functionalities)
   - [How to Use](#how-to-use)
6. [Getting Started](#getting-started)
7. [Containerization with Apptainer (Coming Soon)](#containerization-with-apptainer-coming-soon)
8. [Lessons from Tinkering with PyTorch DDP on SLURM](#lessons-from-tinkering-with-pytorch-ddp-on-slurm)
   - [Key Notes](#key-notes)
   - [Useful SLURM Commands](#useful-slurm-commands)
9. [Contributing](#contributing)
10. [License](#license)
11. [Acknowledgments](#acknowledgments)
 
---

## Quick Start Guide

1. **Clone and Navigate**:
   ```bash
   git clone https://github.com/albertbuchard/slurm-pytorch-ddp-boilerplate.git
   cd slurm-pytorch-ddp-boilerplate
   ```

2. **Local Environment Setup**:

   - Linux/Mac:
     ```bash
     ./setup_environment.sh cpu # or gpu
     ```
   - Windows:
     ```bash
     setup_environment.bat cpu # or gpu
     ```
     
    Note: these scripts will create a virtual environment at `$HOME/venv/slurm-pytorch-ddp-boilerplate`. 

3. **Wandb Setup**:
   - Rename `.env.template` to `.env`.
   - Add your `WANDB_API_KEY` from [wandb's authorization page](https://wandb.ai/authorize).

4. **Run Locally**:
   ```bash
   # Ensure you are in the virtual environment
   source $HOME/venv/slurm-pytorch-ddp-boilerplate/bin/activate
   # For single task training
   python main.py 
   # For locally distributed training
   torchrun --nproc_per_node=2 main.py --cpu # for CPU training
   ```

   Note: Add any required arguments as needed, see [command-lines arguments](#command-line-arguments).

5. **SLURM Users**: 

   - Fork the git repo, and modify SLURM scripts in the `slurm` directory to ensure that the configuration match your cluster's resources and modules.
   - Connect to your SLURM cluster, and clone your forked repo.
   - Set the .env file with your wandb API key.
   - Submit your job using `sbatch`.

      ```bash
      cd slurm
      sbatch slurm_train.sh
      ```
 

That's it! Dive into the detailed sections for configurations, integrations, and advanced usage.

--- 

## Features

- [x] **MNIST DDP Example**: DDP solution to a simple MNIST classification task to demonstrate the boilerplate's capabilities. (see [src/trainer_v1](https://github.com/albertbuchard/slurm-pytorch-ddp-boilerplate/tree/main/src/trainer_v1), adapted from [this repo](https://github.com/pytorch/examples/blob/main/mnist/main.py))
- [x] **Configuration Management**: `CurrentConfig` is singleton pattern for configuration management ensuring consistent configuration usage across the codebase.
- [x] **SLURM Integration**: Ready-made SLURM scripts to deploy your deep learning jobs on HPC clusters. (see `slurm`)
  - **Slurm Setup Script**: SLURM configuration scripts needs to be adapted to your cluster module. 
- [x] **DDP Utilities**: 
  - **DDP Identity**: A singleton pattern for DDP identity management ensuring consistent process usage across DDP.
  - **DDP Iterable Datasets**: Example iterable dataset tailored for DDP to handle data sharding and distribution.
  - **Device Management**: A singleton pattern for device management ensuring consistent device usage across DDP.
- [x] **Weights and Biases Support**: A DDP wrapper for wandb to enable distributed experiment logging and sweeps.
- [x] **Easy environment Setup**: Simplified environment setup scripts for both Linux/Mac and Windows.
- [ ] **Apptainer Support**: Containerize your deep learning environments with Apptainer.
- [ ] **Basic Testing**: Basic testing setup to ensure the functionality of your models and scripts.
 
---

## Installation

### Prerequisites
- **HPC Cluster**: Ensure you have access to a High-Performance Computing (HPC) cluster with SLURM installed.
- **Local Execution**:
  - Python 3.8 or higher. You can check your version using `python --version`.
  - CUDA support if you're planning to use GPUs for training.
    - Check out [PyTorch's local installation guide](https://pytorch.org/get-started/locally/) for instructions.
    - Refer to [NVIDIA's CUDA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) for additional details.

### Setup Steps

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/albertbuchard/slurm-pytorch-ddp-boilerplate.git
   cd slurm-pytorch-ddp-boilerplate
   ```

2. **Environment Setup**:

   - For **Linux/Mac**:
     ```bash
     ./setup_environment.sh
     ```

   - For **Windows**:
     ```bash
     setup_environment.bat
     ```

3. **Configure Weights & Biases**:

   - Rename `.env.template` to `.env`.
   - Add your `WANDB_API_KEY`. If you don't have an API key yet, obtain it from the [wandb authorization page](https://wandb.ai/authorize).

---

## Usage

1. **Training Code**:
   - Navigate to the `src/trainer_v1` directory.
   - Here, you'll find the MNISTNet and trainer code tailored for an MNIST classification task.
   - Feel free to modify the code based on your project's requirements.

2. **Script Execution**:
   - `main.py` acts as the primary entry point for the boilerplate. 
   - You can either use it directly or adapt its structure to better suit your project's needs.

3. **HPC Cluster Deployment**:
   - If you're planning to deploy your models on an HPC cluster, head over to the `slurm` directory.
   - Adjust the provided SLURM scripts to match your cluster's configurations.
   - Submit your deep learning jobs using the `sbatch` command.
   - **Note**: You'll likely need to customize the slurm job script, ensuring it aligns with your cluster's available modules. As of now, it uses the following modules:
     - GCCcore/11.3.0
     - Python 3.10.4
     - NCCL/2.12.12-CUDA-11.7.0
     - PyTorch 2.0.1 (please note that the PyTorch version is not hardcoded in the job scripts; ensure it matches your requirements).

--- 

## Main.py Functionality

The `main.py` serves as the primary entry point for the boilerplate and offers a rich set of configurable parameters for users to customize their distributed training and logging experience. Here's what you can do with it:

### Command-Line Arguments

- **Training Configuration**:
  - `--batch-size`: Specify the input batch size for training.
  - `--test-batch-size`: Define the input batch size for testing.
  - `--epochs`: Set the number of epochs for training.
  - `--dry-run`: Quickly validate a single pass through the data.
  - `--seed`: Set a random seed for reproducibility.

- **Distributed Data Parallel (DDP) Configuration**:
  - `--cpu`: Use CPU for training, disabling CUDA/MPS.
  - `--backend`: Specify the distributed backend but should be detected automatically (gloo CPU, nccl for CUDA).

- **Weights & Biases (wandb) Configuration**:
  - `--no-wandb`: Disable wandb logging.
  - `--wandb-all-processes`: Enable wandb logging on all processes.
  - `--wandb-project`: Define the wandb project name.
  - `--wandb-group`: Set the wandb group.
  - `--sweep-config`: Provide a path to the wandb sweep configuration.

- **Training Modes**:
  - `--sweep`: Use this flag to run a wandb sweep.

### Core Functionalities

- **Distributed Training Setup**: The code initializes distributed training based on the provided configurations. It ensures that the appropriate device (CPU/GPU/MPS) is used and sets up the distributed backend accordingly.

- **Weights & Biases Integration**: The code supports Weights & Biases for experiment tracking and logging. Users can also conduct hyperparameter sweeps using wandb. See `src/ddp/wandb_ddp.py` for more details.

- **Training Loop**: automatically detects if the run or `--sweep` should be distributed or not.

- **Exception Handling and Cleanup**: Any exceptions during execution are captured and printed, ensuring that users are informed of any issues. The code also measures the total execution time and cleans up distributed resources at the end.

---

## Getting Started

1. **Configuration**: Before running `main.py`, ensure you've set up your configurations correctly. The code uses the `current_config` singleton to store and manage various configurations, including model, data, trainer, wandb, and sweep configurations.

2. **Running the Script**: After setting up your environment and configurations:

   ```bash
   python main.py [YOUR_ARGUMENTS_HERE]
   ```

   Replace `[YOUR_ARGUMENTS_HERE]` with any of the command-line arguments mentioned above to customize your training and logging experience.

3. **Distributed SLURM Training**: The boilerplate is designed for distributed training. Use SLURM or your preferred job scheduler to deploy the script across multiple nodes or GPUs.

4. **Distributed Local Training**: The boilerplate also supports distributed training on a single machine. You can use the `torchrun` or `torch.distributed.launch` utilities to run the script on multiple GPUs.

    ```bash
    torchrun --nproc_per_node=2 main.py --cpu # for CPU training
    torchrun --nproc_per_node=2 main.py # for GPU training (requires CUDA and two GPUs)
    ```

---

## Wandb DDP Wrapper: DistributedWandb

`DistributedWandb` is a singleton class designed to manage Weights & Biases (wandb) logging in a distributed setting. Here's a detailed look:

### Key Features:

- **Selective Logging**: By default, only the primary (rank 0) process emits to wandb. This prevents redundant logging from all processes. However, users can choose to enable logging from all processes.

- **Wandb Sweep and Agent Integration**: The class supports wandb sweeps, allowing hyperparameter optimization in a distributed manner. 

- **Dynamic Initialization**: Users can dynamically reinitialize wandb within their script, updating project or group details.

- **Fallback Mechanism**: When wandb logging is disabled, the class ensures that the calls to wandb methods are safely ignored, facilitating a smooth transition between logging and non-logging modes.

### Methods and Functionalities:

- **setup**: Initializes wandb based on the given configurations.

- **init**: Reinitializes wandb. Useful when you need to update certain configurations during runtime.

- **finish**: Closes the current wandb run.

- **sweep**: Creates a new wandb sweep and returns its ID.

- **agent**: Starts a wandb agent for the given sweep, allowing distributed hyperparameter optimization.

- **log**: Logs data to wandb.

- **watch**: Allows monitoring of PyTorch models with wandb.

- **every_process**: A property that determines if every process should log to wandb.

- **Dynamic Attribute Access**: The class can dynamically access and set attributes of the underlying wandb module, ensuring a seamless and intuitive user experience.

### How to Use:

1. Instantiate the `DistributedWandb` class:

   ```python
   wandb = DistributedWandb(every_process=False, project="Your_Project_Name")
   ```

2. Log data:

   ```python
   wandb.log({"loss": loss_value})
   ```

3. Use wandb's functionalities as you would in a non-distributed setting. The `DistributedWandb` class ensures that everything works seamlessly in the background.

---

The `DistributedWandb` class, combined with the rest of the boilerplate, provides a holistic framework for distributed deep learning experiments, making it easier for practitioners to scale their projects and achieve reproducibility.

**Note**: Users should be familiar with the basics of Weights & Biases to make the most out of the `DistributedWandb` utility. Adjustments might be needed based on specific project requirements.

---  

## Containerization with Apptainer (Coming Soon)

Use the provided `apptainer.def` and `apptainer_util.sh` to containerize your deep learning environment, ensuring consistent reproducibility across different platforms.

Sure! I've integrated your notes into a new section titled "Lessons from Tinkering with PyTorch DDP on SLURM" for the README.md:

---

## Lessons from Tinkering with PyTorch DDP on SLURM

During the development of this boilerplate, various learnings emerged that could be beneficial for others embarking on a similar journey. Here are some key takeaways:

### Key Notes:

- **Avoid `torchrun` On Slurm**: It occasionally spawns double the tasks on a single node. Could not find a solution to this issue.
- **Explicit Module Loading in `.sbatch` Files**: It seems you cannot call a bash script to load the slurm modules. You need to explicitly load them in the `.sbatch` file with the `module load` command.
- **gpu-per-task vs gres**: Using `gres` allows the utilization of `local_rank`. Learn more [here](https://stackoverflow.com/questions/67091056/gpu-allocation-in-slurm-gres-vs-gpus-per-task-and-mpirun-vs-srun).
- **Do Not Use Job Arrays**: SLURM job arrays are meant for independent jobs. These will usually not run concurrently, which isn't handled by `torch.distributed`.
- **DDP & Data Sharding**: DDP primarily focuses on data sharding using DistributedDataSampler. If you have an `IterativeDataset`, you'd need custom logic in `__iter__()`. [DDPIterableDataset](https://github.com/albertbuchard/slurm-pytorch-ddp-boilerplate/tree/main/ddp/ddp_iterable_dataset.py) is an example of such implementation, but approaches like `webdataset` or `torchdata DataLoader2` are probably bests (coming soon).
- **Join Context Manager**: the Join() context manager is necessary to ensure that all processes are synchronized before a dataloading operation where the data is sharded with potentially different sizes. Without it, the processes will hang indefinitely. The boilerplate uses the `conditional_join(model, optimizer)` context manager to ensure that the join is only called when DDP is enabled.

### Useful SLURM Commands:

- Monitor job: `watch -n 1 scontrol show job <jobid>`
- Real-time logs: `tail -f  slurm-<jobid>.out`
- View short logs: `cat slurm-<jobid>.out`
- View long logs: `less slurm-<jobid>.out` 
  - press `q` to exit, `j` to scroll down, `k` to scroll up
  - `G` to go to the end, `gg` to go to the beginning
- Account information: `sacct`
- User-specific queue: `squeue -u <user>`

---

## Contributing

Feel free to fork this repository, make your changes, and submit pull requests. We appreciate any contributions that can enhance the capabilities of this boilerplate!

---

## License

This project is open-source and available under the MIT License.

---

## Acknowledgments

Thank you to the PyTorch team for their robust deep learning framework, to the SLURM team for their powerful job scheduler, and to Weights & Biases for their comprehensive experiment tracking platform.

---