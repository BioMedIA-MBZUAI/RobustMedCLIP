import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from cls_to_names import *
from sklearn.model_selection import StratifiedShuffleSplit
import medmnist

DATA_COLLECTIONS = {
    'MedMNIST': {
        "root": "./data/MedMNIST-C",
        "datasets": ["bloodmnist", "retinamnist", "breastmnist", "octmnist", "pneumoniamnist"]
    },
    'MediMeta': {
        "root": "./data/MediMeta-C",
        "datasets": ["aml", "fundus", "mammo_calc", "mammo_mass", "pneumonia", "oct", "pbc"]
    }
}

class MedDataset(Dataset):
    """
    Args:
        root (str): Path to the dataset root directory.
        dataset_name (str): Name of the dataset (e.g., 'bloodmnist', 'retinamnist').
        corruption (str, optional): Type of corruption. Set to None for clean dataset.
        severity (int, optional): Severity level of corruption (1-5). Only used if corruption is not None.
        transform (callable, optional): Transform to apply to images.
        split (str, optional): Dataset split ('train', 'val', 'test'). Defaults to 'test'.
        fewshot (float, optional): If provided (between 0 and 1), samples that percentage of data using stratified sampling.
    """
    
    def __init__(self, col, dataset_name, corruption=None, severity=None, transform=None, split='test', fewshot=None):
        
        self.root = DATA_COLLECTIONS[col]['root']
        self.dataset_name = dataset_name
        self.corruption = corruption
        self.severity = severity
        self.transform = transform
        self.split = split
        self.fewshot = fewshot
        
        # Set proper file path based on parameters
        if corruption is None or corruption == 'clean':
            # For clean dataset
            file_path = os.path.join(self.root, dataset_name, split,  'clean.npz')
        else:
            # For corrupted dataset
            if severity is None or not (1 <= severity <= 5):
                raise ValueError("Severity must be an integer between 1 and 5 for corrupted datasets")
            
            corruption_filename = f"{corruption.lower().replace(' ', '_')}_severity_{severity}.npz"
        
            file_path = os.path.join(self.root, dataset_name, split, corruption_filename)
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found at {file_path}")

        # Load dataset
        npz_file = np.load(file_path, mmap_mode="r")
        self.imgs = npz_file["images"]
        self.labels = npz_file["labels"]
        
        # Check if grayscale or RGB
        self.n_channels = 3 if len(self.imgs.shape) == 4 and self.imgs.shape[-1] == 3 else 1
        
        # Default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.Lambda(lambda image: image.convert('RGB')),
                transforms.ToTensor()
            ])
            
        # If fewshot is provided, sample a subset of the data using stratified sampling
        self.indices = np.arange(len(self.imgs))
        if fewshot is not None and 0 < fewshot < 1:
            try:
                # Use StratifiedShuffleSplit to maintain class distribution
                sss = StratifiedShuffleSplit(n_splits=1, test_size=None, train_size=fewshot, random_state=42)
                # Get the indices for the sampled subset
                for train_idx, _ in sss.split(self.indices, self.labels.flatten()):
                    self.indices = train_idx
                print(f"Few-shot sampling: using {len(self.indices)}/{len(self.imgs)} samples ({fewshot*100:.1f}%)")
            except Exception as e:
                print(f"Warning: Failed to perform stratified sampling for {dataset_name}: {e}. Using random sampling instead.")
                # Fall back to random sampling if stratified sampling fails
                np.random.seed(42)
                sample_size = int(len(self.imgs) * fewshot)
                self.indices = np.random.choice(len(self.imgs), size=sample_size, replace=False)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        """
        Returns:
            img (tensor): Image loaded and transformed.
            target (tensor): Corresponding label.
        """
        actual_index = self.indices[index]
        img, target = self.imgs[actual_index], self.labels[actual_index].astype(int)
        img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)
        
        return img, torch.tensor(target)
    
    @staticmethod
    def get_available_corruptions():
        """Returns a list of available corruption types."""
        return [
            'Gaussian Noise',
            'Shot Noise',
            'Impulse Noise',
            'Defocus Blur',
            'Glass Blur',
            'Motion Blur',
            'Zoom Blur',
            'Brightness',
            'Contrast',
            'Pixelate',
            'JPEG'
        ]
    
    
    def montage(self, length=10, replace=False, save_folder=None):
        """
        Create a montage of randomly selected images.

        Args:
            length (int): Number of images per row and column (default=10).
            replace (bool): Whether to allow selecting the same image multiple times.
            save_folder (str, optional): If provided, saves the montage image.

        Returns:
            PIL.Image: The generated montage.
        """
        try:
            # Try to import from medmnist
            from medmnist.utils import montage2d
        except ImportError:
            # If not available, define a simple montage function
            def montage2d(imgs, n_channels, sel):
                # Create a simple grid of images
                grid_size = int(np.sqrt(len(sel)))
                h, w = imgs[0].shape[:2]
                
                if n_channels == 1:
                    montage = np.zeros((grid_size * h, grid_size * w), dtype=np.uint8)
                    for i, idx in enumerate(sel):
                        if i >= grid_size * grid_size:
                            break
                        r, c = i // grid_size, i % grid_size
                        montage[r*h:(r+1)*h, c*w:(c+1)*w] = imgs[idx]
                    
                    # Convert to PIL image
                    montage_img = Image.fromarray(montage)
                else:
                    montage = np.zeros((grid_size * h, grid_size * w, 3), dtype=np.uint8)
                    for i, idx in enumerate(sel):
                        if i >= grid_size * grid_size:
                            break
                        r, c = i // grid_size, i % grid_size
                        montage[r*h:(r+1)*h, c*w:(c+1)*w] = imgs[idx]
                    
                    # Convert to PIL image
                    montage_img = Image.fromarray(montage)
                
                return montage_img
            
        n_sel = length * length  # Total images in montage
        indices = np.arange(n_sel) % len(self)

        # Generate montage
        montage_img = montage2d(imgs=self.imgs, n_channels=self.n_channels, sel=indices)

        # Save montage if required
        if save_folder:
            os.makedirs(save_folder, exist_ok=True)
            corruption_str = f"_{self.corruption}_sev{self.severity}" if self.corruption else "_clean"
            save_path = os.path.join(save_folder, f"montage_{self.dataset_name}{corruption_str}.jpg")
            montage_img.save(save_path)
            print(f"Montage saved at {save_path}")

        return montage_img

class FinetuneDataset(Dataset):
    """
    Dataset class specifically designed for fine-tuning the RobustMedClip model with knowledge distillation.
    
    Args:
        root (str): Path to the dataset root directory.
        dataset_names (list or str): Name(s) of the dataset(s) (e.g., ['bloodmnist', 'retinamnist']).
        class_prompts (dict, optional): Dictionary mapping class indices to text prompts.
        transform (callable, optional): Transform to apply to images.
        split (str, optional): Dataset split ('train', 'val', 'test'). Defaults to 'train'.
        use_corruptions (bool): Whether to include corrupted versions of the data for fine-tuning robustness.
        severity_range (tuple): Range of corruption severities to include (min, max).
        corruption_types (list, str, or int): Specific corruption types to use, or number to randomly select.
        fewshot (float, optional): If provided (between 0 and 1), samples that percentage of data using stratified sampling.
        teacher_transforms (callable, optional): Transform to apply to images for the teacher model.
    """
    
    def __init__(self, dataset_names, class_prompts=None, transform=None, split='val',
                 use_corruptions=True, severity_range=(1, 3), corruption_types=None, fewshot=None, teacher_transforms=None):

        self.transform = transform
        self.teacher_transform = teacher_transforms
        self.split = split
        
        # Handle single dataset name as a string
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        self.dataset_names = dataset_names
        
        # Initialize empty lists to store datasets and their information
        self.datasets = []
        self.dataset_lengths = []
        self.dataset_to_global_label = {}
        self.global_label_count = 0
        
        # Load each clean dataset
        for dataset_name in dataset_names:
            # find the dataset in DATA_COLLECTIONS and get the collection name
            collection_name = self.find_collection(dataset_name)
            
            if collection_name is None:
                raise ValueError(f"Dataset {dataset_name} not found in known collections.")

            # Load clean dataset for this dataset_name
            clean_dataset = MedDataset(
                col=collection_name,
                dataset_name=dataset_name,
                corruption=None,
                transform=transform,
                split=split,
                fewshot=fewshot
            )
            
            self.datasets.append(clean_dataset)
            self.dataset_lengths.append(len(clean_dataset))
            
            # Create a mapping from local dataset labels to global labels
            unique_labels = np.unique(clean_dataset.labels)
            label_mapping = {int(label): self.global_label_count + i for i, label in enumerate(unique_labels)}
            self.dataset_to_global_label[dataset_name] = label_mapping
            self.global_label_count += len(unique_labels)
        
        # Total dataset length
        self.length = sum(self.dataset_lengths)
        
        # Compute dataset offsets for indexing
        self.dataset_offsets = [0]
        for i in range(len(self.dataset_lengths)):
            self.dataset_offsets.append(self.dataset_offsets[-1] + self.dataset_lengths[i])
            
        # Set up text prompts for each class
        self.setup_class_prompts(class_prompts)
    
    @staticmethod
    def find_collection(dataset_name):
        """Find the collection name for a given dataset name"""
        for collection_name, info in DATA_COLLECTIONS.items():
            if dataset_name in info['datasets']:
                return collection_name
        return None
    
    def setup_class_prompts(self, class_prompts=None):
        """Set up text prompts for each class using class names from cls_to_names module"""
        if class_prompts is None:
            self.class_prompts = {}
            
            # Process each dataset to get class names
            for dataset_name in self.dataset_names:
                try:
                    class_names = eval(f"{dataset_name}_classes")
                    
                    # Get the local-to-global label mapping for this dataset
                    local_to_global = self.dataset_to_global_label[dataset_name]
                    
                    # Create prompts using class names
                    for local_label, global_label in local_to_global.items():
                        if 0 <= local_label < len(class_names):
                            class_name = class_names[local_label]
                            # Create a more detailed medical prompt using the class name
                            self.class_prompts[global_label] = f"a medical image of {class_name} belonging to {dataset_name}"
                        else:
                            self.class_prompts[global_label] = f"a medical image showing class {global_label}"
                
                except (NameError, IndexError) as e:
                    # NameError occurs if the class list doesn't exist
                    # IndexError occurs if the label index exceeds the class list length
                    print(f"Warning: Could not load class names for {dataset_name}: {e}")
                    
                    # Try to fall back to cls_to_name dictionary format
                    try:
                        class_mapping_name = f"{dataset_name}_cls_to_name"
                        class_mapping = eval(class_mapping_name)
                        
                        local_to_global = self.dataset_to_global_label[dataset_name]
                        for local_label, global_label in local_to_global.items():
                            if local_label in class_mapping:
                                class_name = class_mapping[local_label]
                                self.class_prompts[global_label] = f"a medical image of {class_name}"
                            else:
                                self.class_prompts[global_label] = f"a medical image showing class {global_label}"
                    
                    except (NameError, TypeError) as e2:
                        # Fallback to generic prompts if no class names are available
                        print(f"  Additionally, could not find dictionary mapping: {e2}")
                        local_to_global = self.dataset_to_global_label[dataset_name]
                        for local_label, global_label in local_to_global.items():
                            self.class_prompts[global_label] = f"a medical image showing class {global_label}"
            
            # Fill in any missing prompts with default format
            for i in range(self.global_label_count):
                if i not in self.class_prompts:
                    self.class_prompts[i] = f"a medical image showing class {i}"
        else:
            self.class_prompts = class_prompts
        
        # make text prompts as a list
        self.text_prompts = [self.class_prompts[i] for i in range(self.global_label_count)]

    def __len__(self):
        return self.length
    
    def get_dataset_info(self, idx):
        """Get dataset index and local index for a global index"""
        dataset_idx = 0
        while dataset_idx < len(self.dataset_offsets) - 1 and idx >= self.dataset_offsets[dataset_idx + 1]:
            dataset_idx += 1
        
        local_idx = idx - self.dataset_offsets[dataset_idx]
        return dataset_idx, local_idx
    
    def __getitem__(self, idx):
        """
        Returns:
            img (tensor): Image loaded and transformed.
            text_prompt (str): Text prompt for the image's class.
            label (tensor): Corresponding class label (global index).
        """
        # Find which dataset the index belongs to
        dataset_idx, local_idx = self.get_dataset_info(idx)
        dataset = self.datasets[dataset_idx]
        
        # Get the item from the corresponding dataset
        img, local_label = dataset[local_idx]
        
        # Convert to global label
        local_label_idx = local_label.item() if isinstance(local_label, torch.Tensor) else local_label
        
        # Get original dataset name (needed for mapping to global label)
        # For clean dataset, it's just dataset_name, for corrupted it's the same as the clean one
        original_dataset_name = dataset.dataset_name
        
        # Map local label to global label
        global_label = self.dataset_to_global_label[original_dataset_name].get(
            local_label_idx, local_label_idx)
        
        # Get the text prompt for this class
        text_prompt = self.class_prompts.get(global_label, f"a medical image showing class {global_label}")
        
        return img, text_prompt, torch.tensor(global_label)

class CustomCollator:
    """
    Collate function for FinetuneDataset.
    Handles batching of images, text prompts, and labels.
    """
    def __init__(self, device="cpu"):
        self.device = device
    
    def __call__(self, batch):
        images, text_prompts, labels = zip(*batch)
        
        # Stack images and labels
        images = torch.stack(images)
        labels = torch.stack(labels) if all(isinstance(l, torch.Tensor) for l in labels) else torch.tensor(labels)
        
        # Keep text prompts as list
        text_prompts = list(text_prompts)
        
        return images, text_prompts, labels

class FineTuneDataset_v2(Dataset):
    pass

def get_dataloader(
    datasets=None,
    col=None,
    corruption=None, 
    severity=None, 
    transform=None,
    train_transform=None,
    val_transform=None,
    split='test',
    batch_size=32, 
    shuffle=True, 
    num_workers=4,
    fewshot=None,
    class_prompts=None,
    use_corruptions=False,
    severity_range=(1, 3),
    corruption_types=None,
    val_split='test',
    return_both_loaders=False,
    finetune_mode=False
):
    """
    Unified function to create DataLoader(s) for both single datasets and for fine-tuning scenarios.
    
    Args:
        datasets (str or list): Name of a single dataset or list of dataset names.
        col (str, optional): Collection name. If None, will try to find the appropriate collection.
        corruption (str, optional): Type of corruption. Set to None for clean dataset.
        severity (int, optional): Severity level of corruption (1-5). Only used if corruption is not None.
        transform (callable, optional): Transform to apply to images (used if not in finetune_mode).
        train_transform (callable, optional): Transform for training images (used in finetune_mode).
        val_transform (callable, optional): Transform for validation images (used in finetune_mode).
        split (str): Dataset split ('train', 'val', 'test'). Defaults to 'test'.
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of workers for the DataLoader.
        fewshot (float, optional): If provided (between 0 and 1), samples that percentage of data using stratified sampling.
        class_prompts (dict, optional): Dictionary mapping class indices to text prompts (used in finetune_mode).
        use_corruptions (bool): Whether to include corrupted versions for training (used in finetune_mode).
        severity_range (tuple): Range of corruption severities to include (min, max) (used in finetune_mode).
        corruption_types (list, str, or int): Specific corruption types to use, or number to randomly select (used in finetune_mode).
        val_split (str): Split to use for validation ('val' or 'test') (used in finetune_mode).
        return_both_loaders (bool): If True, returns (train_loader, val_loader) tuple. Used with finetune_mode.
        finetune_mode (bool): If True, uses FinetuneDataset to create loaders for fine-tuning scenarios.
        
    Returns:
        torch.utils.data.DataLoader or tuple: Single DataLoader if return_both_loaders=False, 
        otherwise (train_loader, val_loader) tuple.
    """
    if datasets is None:
        raise ValueError("datasets parameter must be provided (either a single dataset name or a list)")
    
    # Handle the datasets parameter - convert to list if it's a string
    dataset_list = [datasets] if isinstance(datasets, str) else datasets
    
    # Set up default transforms for finetune mode
    if finetune_mode:
        if train_transform is None:
            train_transform = get_transform('train')
        
        if val_transform is None:
            val_transform = get_transform('val')

        # For fine-tuning, we create train and val datasets
        train_dataset = FinetuneDataset(
            dataset_names=dataset_list,
            class_prompts=class_prompts,
            transform=train_transform,
            split='val',
            use_corruptions=use_corruptions,
            severity_range=severity_range,
            corruption_types=corruption_types,
            fewshot=fewshot,
            teacher_transforms=get_transform('teacher')  # Use teacher transforms for training
        )
        
        val_dataset = FinetuneDataset(
            dataset_names=dataset_list,
            class_prompts=class_prompts,
            transform=val_transform,
            split=val_split,
            use_corruptions=False,  # No corruptions for validation
            # fewshot=fewshot
        )
        
        # Create collator
        collator = CustomCollator()
        
        # Create dataloaders
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collator
        )
        
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=1024,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collator
        )
        
        return (train_loader, val_loader)
    
    else:
        # Default transform for non-finetune mode
        transform = get_transform()

        # Standard single dataset mode
        if len(dataset_list) > 1:
            print(f"Warning: Multiple datasets provided ({dataset_list}), but finetune_mode=False. Using only the first dataset.")
        
        dataset_name = dataset_list[0]
            
        # If collection is not specified, try to find it
        if col is None:
            col = FinetuneDataset.find_collection(dataset_name)
            if col is None:
                raise ValueError(f"Dataset {dataset_name} not found in known collections. Please specify the collection.")

        dataset = MedDataset(
            col=col,
            dataset_name=dataset_name,
            corruption=corruption,
            severity=severity,
            transform=transform,
            split=split,
            fewshot=fewshot
        )
        
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
        
        if return_both_loaders:
            return (loader, loader)
        else:
            return loader

def get_transform(split=''):
    """Create CLIP-compatible image transform"""
    if split == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(lambda image: image.convert('RGB')),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    elif split == 'teacher':
        return transforms.Compose([
            transforms.Resize(336, interpolation=transforms.InterpolationMode.BICUBIC),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.Lambda(lambda image: image.convert('RGB')),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    root = './data/MediMeta-C'  
    dataset_names = ['aml', 'fundus']  
    try:
        robust_dataset = FinetuneDataset(
            root=root,
            dataset_names=dataset_names,
            use_corruptions=False,  # Include corruptions for robustness
            fewshot=0.1  # Use 10% of the data
        )
        
        # Display dataset information
        print(f"\nRobustMedClipDataset loaded with {len(robust_dataset)} samples")
        print(f"Contains {len(robust_dataset.datasets)} datasets total")
        print(f"Global class count: {robust_dataset.global_label_count}")
        
        # Display some prompt examples
        print("\nExample text prompts:")
        for i, (global_label, prompt) in enumerate(robust_dataset.class_prompts.items()):
            print(f"  Class {global_label}: '{prompt}'")
            if i >= 5:  # Show only first few examples
                print("  ...")
                break
        
        # Get a few samples from the dataset
        print("\nSample data:")
        for i in range(3):
            img, text_prompt, label = robust_dataset[i]
            print(f"Sample {i}:")
            print(f"  Image shape: {img.shape}")
            print(f"  Text prompt: '{text_prompt}'")
            print(f"  Label: {label.item()}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check if the datasets exist at the specified path.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()