#!/bin/bash
# Script to run fine-tuning with different few-shot percentages using a loop

# Exit immediately if a command exits with a non-zero status
set -e

# Set common parameters
DATASETS="aml fundus mammo_calc mammo_mass pneumonia oct pbc bloodmnist breastmnist octmnist pneumoniamnist retinamnist"
BACKBONE="vit"
EPOCHS=20
BATCH_SIZE=256
LORA_RANK=16
TEMPERATURE=2.0
ALPHA=0.5
LR=1e-4
GPU=5

# Create a timestamp for logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
mkdir -p logs

echo "Starting fine-tuning experiments at $TIMESTAMP"

# Define few-shot percentages to run
FEWSHOT_PERCENTAGES=(0.1 0.3 0.75 1.0)


# Function to handle errors
handle_error() {
    echo "Error occurred at line $1. Exiting..."
    exit 1
}

# Set up a trap to catch errors
trap 'handle_error $LINENO' ERR

# Loop through each percentage
for i in "${!FEWSHOT_PERCENTAGES[@]}"; do
    FEWSHOT=${FEWSHOT_PERCENTAGES[$i]}
    # multiply by 100 to get percentage
    LABEL=$(echo "${FEWSHOT}" | awk '{printf "%.0f", $1 * 100}')
    LABEL="${LABEL}_percent"
    
    echo "Running with ${LABEL} few-shot samples (${FEWSHOT})"
    
    # Build the command based on the few-shot percentage
    CMD="python ../finetune.py \
        --datasets $DATASETS \
        --backbone $BACKBONE \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --gpu $GPU \
        --lora-rank $LORA_RANK \
        --temperature $TEMPERATURE \
        --alpha $ALPHA \
        --lr $LR \
        --output-dir \"./outputs/exp2/${BACKBONE}/fewshot_${LABEL}\""
    
    # Add fewshot parameter except for 100% (1.0)
    if [ "$FEWSHOT" != "1.0" ]; then
        CMD="$CMD --fewshot $FEWSHOT"
    fi
    
    # Execute the command and log output
    # Using set -o pipefail to ensure that if any command in pipeline fails, the script exits
    set -o pipefail
    eval $CMD 2>&1 | tee "logs/fewshot_${LABEL}_${TIMESTAMP}.log"
    set +o pipefail
    
    # Check if the previous command was successful
    if [ $? -ne 0 ]; then
        echo "Error running fine-tuning with ${LABEL} few-shot samples"
        exit 1
    fi
    
    echo "Completed run with ${LABEL} few-shot samples"
done

echo "All fine-tuning experiments completed"
echo "Check logs in the logs directory and results in the outputs directory"

# Print a summary of the experiments
echo "Summary:"
for LABEL in "${FEWSHOT_LABELS[@]}"; do
    echo "${LABEL} training outputs saved to: ./outputs/exp/${BACKBONE}/fewshot_${LABEL}"
done
echo "Logs saved to: logs/fewshot_*_${TIMESTAMP}.log"