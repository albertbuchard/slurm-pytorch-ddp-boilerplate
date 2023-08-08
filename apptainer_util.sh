#!/bin/bash

# Define color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Function to display usage instructions
function display_usage {
    echo -e "${GREEN}Usage: $0 [--build] [--run] --data=/path/to/data --script=script.py [--args=\"arg1 arg2\"]${NC}"
    echo
    echo "Options:"
    echo "  --build               Build the Apptainer image."
    echo "  --run                 Run the script inside the container."
    echo "  --data=/path/to/data  Specify the path to the data directory."
    echo "  --script=script.py    Specify the Python script to run."
    echo "  --args=\"arg1 arg2\"     Specify the arguments to pass to the Python script."
    echo "By default, the script will run if --build is not specified."
}

# Function to build the Apptainer image
function build_image {
    # Define the Singularity definition file
    local def_file=$1

    # Define the output image file
    local img_file=$2

    # Check if the definition file exists
    if [ ! -f "$def_file" ]; then
        echo -e "${RED}Definition file $def_file not found.${NC}"
        exit 1
    fi

    # Build the image
    echo -e "${GREEN}Building Apptainer image...${NC}"
    apptainer build --fakeroot "$img_file" "$def_file"
    echo -e "${GREEN}Apptainer image built.${NC}"
}

# Function to run a Python script with arguments
function run_python {
    # Define the image file
    local img_file=$1

    # Define the script to run
    local script=$2

    # Define the data path
    local data_path=$3

    # Get the rest of the arguments as an array
    shift 3
    local args=("$@")

    # Check if the image file exists
    if [ ! -f "$img_file" ]; then
        echo -e "${RED}Apptainer image $img_file not found.${NC}"
        exit 1
    fi

    # Run the script
    echo -e "${GREEN}Running Python script...${NC}"
    apptainer run --nv -B "$(pwd)":/app -B "$data_path":/data "$img_file" "/app/$script" "${args[@]}"
    echo -e "${GREEN}Python script finished.${NC}"
}

# If no arguments were provided, display usage instructions
if [ $# -eq 0 ]; then
    display_usage
    exit 1
fi

# Parse command line arguments
while [[ $# -gt 0 ]]
do
    key="$1"

    case $key in
        --build)
        build=true
        shift
        ;;
        --run)
        run=true
        shift
        ;;
        --data=*)
        data="${1#*=}"
        shift
        ;;
        --script=*)
        script="${1#*=}"
        shift
        ;;
        --args=*)
        args="${1#*=}"
        shift
        ;;
        --help)
        display_usage
        exit 0
        ;;
        *)
        echo -e "${RED}Unknown option: $1${NC}"
        display_usage
        exit 1
        ;;
    esac
done


# Call the functions
if [ "$build" = true ]; then
    build_image "$(pwd)/apptainer.def" "slurm-pytorch-ddp-boilerplate.sif"
fi

if [ "$run" = true ] || [ "$build" != true ]; then
  # Check that required arguments were provided
    if [ -z "$data" ] || [ -z "$script" ]; then
        echo -e "${RED}Error: --data and --script options are required.${NC}"
        display_usage
        exit 1
    fi
    run_python "slurm-pytorch-ddp-boilerplate.sif" "$script" "$data" "$args"
fi
