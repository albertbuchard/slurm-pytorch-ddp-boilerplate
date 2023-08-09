#!/bin/bash

VENV_PATH="$HOME/venv/slurm-pytorch-ddp-boilerplate"

if [ ! -d "$VENV_PATH" ]; then
    echo "Creating directory: $VENV_PATH"
    mkdir -p "$VENV_PATH"
fi

setup() {
    echo "Creating venv environment in $VENV_PATH.."
    echo "Python version: $(python --version)"
    python -m venv $VENV_PATH
    source $VENV_PATH/bin/activate
    pip install -r ./requirements.txt
}

gpu_setup() {
    setup
    echo "Setting up GPU dependencies for CUDA 11.7..."
    pip install torch torchvision torchaudio
}

cpu_setup() {
    setup
    echo "Setting up CPU dependencies..."
    source $VENV_PATH/bin/activate
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
}

clean() {
    if declare -f deactivate > /dev/null; then
        echo "Deactivating venv environment..."
        deactivate
    fi
    echo "Removing venv environment..."
    rm -rf $VENV_PATH
    echo "Venv environment removed successfully."
}

case $1 in
    gpu)
        gpu_setup
        ;;
    cpu)
        cpu_setup
        ;;
    clean)
        clean
        ;;
    *)
        echo $"Usage: $0 {gpu|cpu|clean}"
        exit 1
esac