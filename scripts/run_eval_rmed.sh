#!/bin/bash

# Default configuration values - modify these as needed
COLLECTION="medimeta" # medmnist, medimeta, all
OUTPUT_DIR="outputs/results-rank-16" 
MODEL="rmedclip"
BACKBONES="vit" # vit, resnet, all
GPU_ID="5"
CORRUPTIONS="all"
DATASETS=""
OVERWRITE="--overwrite"

PRETRAINED_DIR="outputs/exp2"

# parse backbone list split by comma
IFS=',' read -ra BACKBONE_LIST <<< "$BACKBONES"

# Define few-shot percentages to run
FEWSHOT_PERCENTAGES=(0.1 0.3 0.75 1.0)

# Process each backbone sequentially
for backbone in "${BACKBONE_LIST[@]}"; do
    # Loop through each percentage
    for i in "${!FEWSHOT_PERCENTAGES[@]}"; do
        FEWSHOT=${FEWSHOT_PERCENTAGES[$i]}

        LABEL=$(echo "${FEWSHOT}" | awk '{printf "%.0f", $1 * 100}')
        LABEL="${LABEL}_percent"
    

        echo "Evaluating with ${LABEL} few-shot samples (${FEWSHOT}) for model: $MODEL with backbone: $backbone using GPU $GPU_ID"
        PRETRAINED_PATH="${PRETRAINED_DIR}/${backbone}/fewshot_${LABEL}/checkpoints/best_model/model.pth"
        
        CMD="python ../evaluate.py --model $MODEL --backbone $backbone --gpu $GPU_ID --corruptions $CORRUPTIONS --output_dir $OUTPUT_DIR-fewshot-${LABEL} $OVERWRITE --pretrained_path $PRETRAINED_PATH"

        # Add appropriate dataset parameters based on what was provided
        if [ -n "$DATASETS" ]; then
            # Datasets take precedence over collection
            CMD="$CMD --datasets $DATASETS"
        else
            # Otherwise, use the collection
            CMD="$CMD --collection $COLLECTION"
        fi

        # Execute the command
        echo "Executing: $CMD"
        eval $CMD
    done
done

echo "All evaluations completed!"
