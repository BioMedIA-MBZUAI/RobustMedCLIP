import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional
import json
import os
import argparse
from tqdm import tqdm
import sys
import time
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import torch.nn.functional as F

sys.path.append('.')
from dataset import get_dataloader, DATA_COLLECTIONS
from cls_to_names import *
from utils import _MODELS
from models import (
    ClipZeroShot,
    MedclipZeroShot,
    BioMedClipZeroShot,
    UniMedClipZeroShot,
    RobustMedClip,
)

# Define available models

DATASETS = {
    "medmnist": ('../MedMNIST-C', ["bloodmnist", "retinamnist", "breastmnist", "octmnist", "pneumoniamnist"]),
    "medimeta": ('../MediMeta-C', ["aml", "fundus", "mammo_calc", "mammo_mass", "pneumonia", "oct", "pbc"])
}

AVAILABLE_CORRUPTIONS = [
    'clean', 'gaussian_noise', 'impulse_noise', 'motion_blur', 'zoom_blur', 'pixelate',
      'snow', 'frost', 'fog', 'brightness', 'contrast' , 'shot_noise' , 'defocus_blur', 'glass_blur', 
]

SEVERITY_LEVELS = [1, 2, 3, 4, 5]

PROMPT_TEMPLATES = [
    "a medical image of {}",
    "an image of {}",
    "a picture of {}",
    "a photo of a {}",
    "a photograph of {}",
    "a medical image showing {}",
]

def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="Evaluate CLIP models on medical datasets")
    parser.add_argument("--collection", type=str, default="medmnist", choices=["medmnist", "medimeta", "all"], 
                        help="Dataset collection to evaluate")
    parser.add_argument("--datasets", type=str, default=None, 
                        help="Comma-separated list of specific datasets to evaluate (takes precedence over collection)")
    parser.add_argument("--output_dir", type=str, default="../results", 
                        help="Output directory for results")
    parser.add_argument("--model", type=str, required=True, 
                        help="Model to evaluate")
    parser.add_argument("--backbone", type=str, required=True, 
                        help="Backbone to evaluate")
    parser.add_argument("--pretrained_path", type=str, required=True,
                        help="Path to pretrained model weights")
    parser.add_argument("--gpu", type=int, default=0, 
                        help="GPU ID to use")
    parser.add_argument("--corruptions", type=str, default="clean", 
                        help="Comma-separated list of corruptions to evaluate")
    parser.add_argument("--overwrite", action="store_true", 
                        help="Overwrite existing results")
    parser.add_argument("--prompt_template", type=str, default="a medical image of a {} belonging to {}",
                        help="Prompt template for text features")
    
    return parser.parse_args()

def load_model(args, device="cuda"):
    """
    Load a model once to be used for multiple dataset evaluations
    """
    # Initialize the model
    model_class = eval(_MODELS[args.model]['class_name'])
    model = model_class(
        vision_cls=args.backbone,
        device=device,
        lora_rank=args.lora_rank,
        load_pretrained=False,
    )
    if args.pretrained_path:
        print(f"Loading model from {args.pretrained_path}")
        model.load(args.pretrained_path)
    model.to(device)
    return model

def evaluate(args, model, datasets_to_evaluate):
    """
    Unified function to evaluate a model on one or more datasets
    
    Args:
        args: Command line arguments
        model: Pre-loaded model to use for evaluation
        datasets_to_evaluate: List of dataset names to evaluate
    
    Returns:
        Dictionary containing evaluation results
    """
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    
    # Get model information for output paths
    model_name = args.model
    backbone = args.backbone
    
    all_results = {}
    
    for dataset_name in datasets_to_evaluate:
        print(f"Evaluating on {dataset_name} using device {device}")
        
        # Create output directory
        output_dir = os.path.join(args.output_dir, model_name, backbone)
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if results already exist
        result_file = os.path.join(output_dir, f"{dataset_name}_results.json")
        if os.path.exists(result_file) and not args.overwrite:
            print(f"Results already exist for {model_name}/{backbone} on {dataset_name}. Skipping...")
            with open(result_file, "r") as f:
                dataset_results = json.load(f)
                all_results[dataset_name] = dataset_results.get(dataset_name, {})
            continue
        
        # Get class names and compute text features
        class_names = eval(f"{dataset_name}_classes")
        input_text = [args.prompt_template.format(cls, dataset_name) for cls in class_names]
        text_features = model.text_features(input_text)
        
        # Initialize results dictionary for this dataset
        dataset_results = {dataset_name: {}}
        
        # Loop through corruptions and severities
        for corruption in args.corruptions:
            dataset_results[dataset_name][corruption] = {}
            
            # For clean corruption, we only have one severity level
            severity_levels = [1] if corruption == "clean" else SEVERITY_LEVELS
            
            for severity in severity_levels:
                try:
                    dataloader = get_dataloader(
                        datasets=dataset_name,
                        col=None, 
                        corruption=corruption, 
                        severity=severity,
                        transform=None,  # No transforms needed as models handle this
                        batch_size=2048,
                    )
                except FileNotFoundError:
                    print(f"Skipping {corruption} - Severity {severity} for {dataset_name}")
                    continue
                
                all_labels, all_preds = [], []
                for images, labels in tqdm(dataloader, desc=f"Evaluating {dataset_name} - {corruption} - Severity {severity}"):
                    preds = model.batch_predict(images, text_features)
                    all_labels.extend(labels.numpy().reshape(-1))
                    all_preds.extend(preds.cpu().numpy())
                
                all_preds = np.array(all_preds)
                all_labels = np.array(all_labels)
                
                # Calculate accuracy using argmax predictions
                accuracy = accuracy_score(all_labels, np.argmax(all_preds, axis=1))

                # For ROC AUC score calculation
                if len(class_names) > 2:
                    roc_auc = roc_auc_score(all_labels, all_preds, multi_class="ovr")
                else:
                    roc_auc = roc_auc_score(all_labels, all_preds[:, 1])
                
                dataset_results[dataset_name][corruption][severity] = {
                    "accuracy": float(accuracy), 
                    "roc_auc": float(roc_auc)
                }
        
        # Save individual dataset results to file
        with open(result_file, "w") as f:
            json.dump(dataset_results, f, indent=4)
        
        all_results[dataset_name] = dataset_results[dataset_name]
    
    # Save combined results if we evaluated multiple datasets
    if len(datasets_to_evaluate) > 1:
        output_dir = os.path.join(args.output_dir, model_name, backbone)
        with open(os.path.join(output_dir, "all_results.json"), "w") as f:
            json.dump(all_results, f, indent=4)
    
    return all_results

def main():
    args = parse_args()
    
    # Process arguments
    if args.corruptions == "all":
        args.corruptions = AVAILABLE_CORRUPTIONS
    else:
        args.corruptions = args.corruptions.split(",")
    
    # Set environment variable for GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    device = "cpu" if not torch.cuda.is_available() else f"cuda:{args.gpu}"
    print(f"Using GPU: {args.gpu}")
    
    # Validate model and backbone
    if args.model not in _MODELS:
        print(f"Error: Unknown model {args.model}")
        sys.exit(1)
        
    if args.backbone not in _MODELS[args.model]['backbones']:
        print(f"Error: Unknown backbone {args.backbone} for model {args.model}")
        sys.exit(1)
    
    datasets_to_evaluate = []
    
    if args.datasets:
        dataset_list = [d.strip() for d in args.datasets.split(',')]
        for dataset in dataset_list:
            found = False
            for collection in DATA_COLLECTIONS.values():
                if dataset in collection['datasets']:
                    found = True
                    datasets_to_evaluate.append(dataset)
                    break
            if not found:
                print(f"Warning: Unknown dataset {dataset}, skipping...")
    
    else:
        if args.collection == "all":
            for collection in DATA_COLLECTIONS.values():
                datasets_to_evaluate.extend(collection['datasets'])
        else:
            datasets_to_evaluate = DATA_COLLECTIONS[args.collection]['datasets']
    
    if not datasets_to_evaluate:
        print("Error: No valid datasets to evaluate!")
        sys.exit(1)
    
    print(f"Evaluating datasets: {', '.join(datasets_to_evaluate)}")
    args.lora_rank = 16
    model = load_model(args, device)
    evaluate(args, model, datasets_to_evaluate)

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
