#!/bin/bash

# Default configuration values - modify these as needed
COLLECTION="all" # medmnist, medimeta, all
OUTPUT_DIR="/home/raza.imam/Documents/rmedclip/outputs/results/rmedclip/vitb16_10perc" 
MODELS="rmedclip"
BACKBONES="all" # vit, resnet, all
GPU_ID="4"
CORRUPTIONS="all"
DATASETS=""
OVERWRITE="--overwrite"

# Parse models and run sequentially
if [ "$MODELS" = "all" ]; then
  MODEL_LIST=("clip" "medclip" "biomedclip" "unimedclip")
else
  IFS=',' read -ra MODEL_LIST <<< "$MODELS"
fi

# Process each model sequentially
for m in "${MODEL_LIST[@]}"; do
  # Determine backbones for this model
  if [ -z "$BACKBONES" ] || [ "$BACKBONES" = "all" ]; then
    case "$m" in
      "clip")
        BACKBONE_LIST=("vit" "resnet")
        ;;
      "medclip")
        BACKBONE_LIST=("vit" "resnet")
        ;;
      "biomedclip")
        BACKBONE_LIST=("vit")
        ;;
      "unimedclip")
        BACKBONE_LIST=("vit-B-16")
        ;;
      "rmedclip")
        BACKBONE_LIST=("vit" "resnet")
        ;;
      *)
        echo "Unknown model: $m"
        continue
        ;;
    esac
  else
    IFS=',' read -ra BACKBONE_LIST <<< "$BACKBONES"
  fi
  
  # Process each backbone sequentially
  for backbone in "${BACKBONE_LIST[@]}"; do
    echo "Processing model: $m with backbone: $backbone using GPU $GPU_ID"
    
    # Build the command with the appropriate parameters
    CMD="python ../evaluate.py --model $m --backbone $backbone --gpu $GPU_ID --corruptions $CORRUPTIONS --output_dir $OUTPUT_DIR $OVERWRITE"
    
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
