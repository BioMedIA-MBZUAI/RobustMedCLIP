import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import logging
from tqdm import tqdm
import numpy as np
from datetime import datetime
import random
from pathlib import Path
import open_clip

from models import RobustMedClip
from dataset import get_dataloader, get_transform, FinetuneDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Fine-tune RobustMedClip model')
    
    # Dataset arguments
    parser.add_argument('--datasets', nargs='+', default=['aml'], 
                        help='List of datasets to use for fine-tuning')
    parser.add_argument('--use-corruptions', action='store_true', 
                        help='Include corrupted images during training for robustness')
    parser.add_argument('--severity-range', nargs=2, type=int, default=[1, 3], 
                        help='Range of corruption severity (min max)')
    parser.add_argument('--corruption-types', nargs='+', default=None, 
                        help='List of corruption types to use (or integer to select randomly)')
    parser.add_argument('--fewshot', type=float, default=None, 
                        help='Fraction of training data to use (for few-shot learning)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=10, 
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128, 
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, 
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, 
                        help='Weight decay coefficient')
    parser.add_argument('--temperature', type=float, default=2.0, 
                        help='Temperature for knowledge distillation')
    parser.add_argument('--alpha', type=float, default=0.5, 
                        help='Weight for distillation loss (alpha) vs task loss (1-alpha)')
    
    # Model arguments
    parser.add_argument('--backbone', type=str, default='vit', 
                        help='Model architecture name (e.g., ViT-B-32)')
    
    # LoRA arguments
    parser.add_argument('--lora-rank', type=int, default=4, 
                        help='Rank of LoRA adaptation matrices')
    parser.add_argument('--lora-alpha', type=int, default=16, 
                        help='Alpha parameter for LoRA')
    parser.add_argument('--lora-dropout', type=float, default=0.1, 
                        help='Dropout probability for LoRA layers')
    
    # Hardware/software arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU index to use (if multiple GPUs are available)')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    parser.add_argument('--num-workers', type=int, default=4, 
                        help='Number of worker processes for data loading')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='./outputs', 
                        help='Directory to save outputs')
    parser.add_argument('--log-interval', type=int, default=10, 
                        help='Logging interval (batches)')
    parser.add_argument('--save-interval', type=int, default=1, 
                        help='Model saving interval (epochs)')
    parser.add_argument('--eval-interval', type=int, default=1, 
                        help='Evaluation interval (epochs)')
    parser.add_argument('--resume', type=str, default=None, 
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Process corruption_types if it's an integer
    if args.corruption_types and len(args.corruption_types) == 1:
        try:
            args.corruption_types = int(args.corruption_types[0])
        except ValueError:
            pass
    
    return args

def evaluate(model, val_loader, device):
    """Evaluate the model on the validation set"""
    model.eval()
    correct = 0
    total = 0
    text_prompts = val_loader.dataset.text_prompts
    text_features = model.text_features(text_prompts)
    with torch.no_grad():
        for images, _, labels in tqdm(val_loader, desc="Evaluating"):
            # Move tensors to device with explicit dtype
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device)
        
            # Get predictions with error handling
            logits = model.batch_predict(images, text_features)
            predictions = torch.argmax(logits, dim=1)
                
            # Update statistics
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
                
    # Avoid division by zero
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return accuracy

def train_epoch(model, train_loader, text_features, optimizer, epoch, args):
    """Train the model for one epoch"""
    # Ensure model components that should be trainable are in train mode
    model.model.train()
    
    # Set up loss functions
    task_criterion = nn.CrossEntropyLoss(ignore_index=-100)  # Use ignore_index for invalid labels
    
    running_task_loss = 0.0

    for batch_idx, (images, _, labels) in enumerate(train_loader):
        try:
            images = images.to(args.device)
            labels = labels.to(args.device)
            
            optimizer.zero_grad()
            image_features = model.model.encode_image(images)

            logits = image_features @ text_features.T
            
            # Calculate tasks losses
            task_loss = task_criterion(logits, labels)
            
            # Backpropagation
            try:
                task_loss.backward()
                optimizer.step()
            except RuntimeError as e:
                logger.error(f"Error during backward pass: {e}")
                continue
            
            running_task_loss += task_loss.item()
            
            # Print statistics
            if batch_idx % args.log_interval == 0:
                logger.info(
                    f"Train Epoch: {epoch} [{batch_idx}/{len(train_loader)} "
                    f"({100. * batch_idx / len(train_loader):.0f}%)]\t"
                    f"Loss: {task_loss.item():.4f}\t"
                )
        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {str(e)}")
            continue
    
    avg_task_loss = running_task_loss / len(train_loader)
    
    return {
        'loss': avg_task_loss,
    }

def save_checkpoint(model, optimizer, epoch, args, metrics, best=False):
    """Save model checkpoint"""
    checkpoint_dir = Path(args.output_dir) / "checkpoints"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    if best:
        save_path = checkpoint_dir / f"best_model"
    else:
        save_path = checkpoint_dir / f"checkpoint_last"

    save_path.mkdir(parents=True, exist_ok=True)

    model.save(save_path)
    torch.save({
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'metrics': metrics,
        'args': vars(args)
    }, save_path / f"checkpoint.pth")
    
    logger.info(f"Checkpoint saved to {save_path}")

def main():
    """Main function for fine-tuning RobustMedClip"""
    args = parse_args()
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print (f"Using device: {args.device}")
    
    # Set up logging to file
    log_file = Path(args.output_dir) / f"training_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"Arguments: {args}")
    
    # Create data loaders
    logger.info(f"Creating dataloaders for datasets: {args.datasets}")
    train_loader, val_loader = get_dataloader(
        datasets=args.datasets,
        use_corruptions=args.use_corruptions,
        severity_range=tuple(args.severity_range),
        corruption_types=args.corruption_types,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        fewshot=args.fewshot,
        finetune_mode=True
    )
    
    # Create model
    logger.info("Initializing RobustMedClip model")
    model = RobustMedClip(
        vision_cls=args.backbone,
        device=args.device,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    model.model.to(args.device)

    text_features = model.text_features(train_loader.dataset.text_prompts)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_acc = 0.0
    
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume / "checkpoint.pth", map_location=args.device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.load(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        if 'metrics' in checkpoint and 'best_acc' in checkpoint['metrics']:
            best_acc = checkpoint['metrics']['best_acc']
    
    # Training loop
    logger.info("Starting training")
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, text_features, optimizer, epoch+1, args)
        logger.info(f"Epoch {epoch+1} - Train Loss: {train_metrics['loss']:.4f}")
        
        # Evaluate with error handling
        if (epoch + 1) % args.eval_interval == 0:
            try:
                # Clear CUDA cache before evaluation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                val_acc = evaluate(model, val_loader, args.device)
                logger.info(f"Epoch {epoch+1} - Validation Accuracy: {val_acc:.2f}%")
                
                # Save best model
                if val_acc > best_acc:
                    best_acc = val_acc
                    logger.info(f"New best validation accuracy: {best_acc:.2f}%")
                    save_checkpoint(model, optimizer, epoch+1, args, {
                        'val_acc': val_acc,
                        'best_acc': best_acc,
                        **train_metrics
                    }, best=True)
            except Exception as e:
                logger.error(f"Error during evaluation in epoch {epoch+1}: {e}")
                logger.info("Skipping evaluation for this epoch")
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(model, optimizer, epoch+1, args, {
                'val_acc': val_acc if (epoch + 1) % args.eval_interval == 0 else None,
                'best_acc': best_acc,
                **train_metrics
            })
    
    # Final evaluation
    final_acc = evaluate(model, val_loader, args.device)
    logger.info(f"Final validation accuracy: {final_acc:.2f}%")
    logger.info(f"Best validation accuracy: {best_acc:.2f}%")
    
    # Save final model
    save_checkpoint(model, optimizer, args.epochs, args, {
        'val_acc': final_acc,
        'best_acc': best_acc
    }, best=False)
    
    logger.info("Training completed")

if __name__ == "__main__":
    main()
