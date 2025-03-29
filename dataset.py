import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class MedDataset(Dataset):
    """
    Args:
        root (str): Path to the dataset root directory.
        dataset_name (str): Name of the dataset (e.g., 'bloodmnist', 'retinamnist').
        corruption (str, optional): Type of corruption. Set to None for clean dataset.
        severity (int, optional): Severity level of corruption (1-5). Only used if corruption is not None.
        transform (callable, optional): Transform to apply to images.
        split (str, optional): Dataset split ('train', 'val', 'test'). Defaults to 'test'.
    """
    
    def __init__(self, root, dataset_name, corruption=None, severity=None, transform=None, split='test'):
        self.root = root
        self.dataset_name = dataset_name
        self.corruption = corruption
        self.severity = severity
        self.transform = transform
        self.split = split
        
        # Set proper file path based on parameters
        if corruption is None or corruption == 'clean':
            # For clean dataset
            file_path = os.path.join(root, dataset_name, split,  'clean.npz')
        else:
            # For corrupted dataset
            if severity is None or not (1 <= severity <= 5):
                raise ValueError("Severity must be an integer between 1 and 5 for corrupted datasets")
            
            corruption_filename = f"{corruption.lower().replace(' ', '_')}_severity_{severity}.npz"
        
            file_path = os.path.join(root, dataset_name, split, corruption_filename)
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
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        """
        Returns:
            img (tensor): Image loaded and transformed.
            target (tensor): Corresponding label.
        """
        img, target = self.imgs[index], self.labels[index].astype(int)
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

class ConcatenatedMedDataset(Dataset):
    """Dataset that concatenates multiple MedMNIST datasets with global labels"""
    
    def __init__(self, root, dataset_names, transform=None, split='train'):
        self.root = root
        self.dataset_names = dataset_names
        self.transform = transform
        self.split = split
        
        self.datasets = []
        self.class_maps = {}
        self.global_to_local = {}
        
        current_class_idx = 0
        
        # Load each dataset and build global class mapping
        for dataset_name in dataset_names:
            dataset = MedMNISTCDataset(
                root=root,
                dataset_name=dataset_name,
                corruption=None,  # Use clean dataset for training
                transform=transform,
                split=split
            )
            
            self.datasets.append(dataset)
            
            # Find the number of classes in this dataset
            if hasattr(dataset, 'classes'):
                num_classes = len(dataset.classes)
            else:
                # Infer from labels
                num_classes = max(dataset.labels.flatten()) + 1
            
            # Create mapping from global to local class indices
            dataset_map = {}
            for i in range(num_classes):
                dataset_map[current_class_idx] = i
                current_class_idx += 1
            
            self.class_maps[dataset_name] = dataset_map
            
            # Reverse mapping: local class in specific dataset -> global class
            for global_idx, local_idx in dataset_map.items():
                self.global_to_local[(dataset_name, local_idx)] = global_idx
        
        # Total number of classes across all datasets
        self.num_classes = current_class_idx
        
        # Compute dataset lengths
        self.dataset_lengths = [len(dataset) for dataset in self.datasets]
        self.length = sum(self.dataset_lengths)
        
        # Compute dataset offsets
        self.dataset_offsets = [0]
        for i in range(len(self.dataset_lengths)):
            self.dataset_offsets.append(self.dataset_offsets[-1] + self.dataset_lengths[i])
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Find which dataset the index belongs to
        dataset_idx = 0
        while dataset_idx < len(self.dataset_offsets) - 1 and idx >= self.dataset_offsets[dataset_idx + 1]:
            dataset_idx += 1
        
        # Get the item from the corresponding dataset
        local_idx = idx - self.dataset_offsets[dataset_idx]
        img, local_label = self.datasets[dataset_idx][local_idx]
        
        # Convert local label to global label
        dataset_name = self.dataset_names[dataset_idx]
        local_label_idx = local_label.item() if isinstance(local_label, torch.Tensor) else local_label
        global_label = self.global_to_local[(dataset_name, local_label_idx)]
        
        return img, torch.tensor(global_label)

def get_transform():
    """Create CLIP-compatible image transform"""
    return transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.Lambda(lambda image: image.convert('RGB')),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
  
def get_dataloader(
    root, 
    dataset_name, 
    corruption=None, 
    severity=None, 
    transform=None, 
    split='test',
    batch_size=32, 
    shuffle=True, 
    num_workers=4
):

    dataset = MedDataset(
        root=root,
        dataset_name=dataset_name,
        corruption=corruption,
        severity=severity,
        transform=transform,
        split=split
    )
    
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )


# Example usage
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    
    root = '../MediMeta-C'  
    dataset_name = 'fundus'  
    
    # Try to load a clean dataset
    try:
        clean_dataset = MedDataset(
            root=root,
            dataset_name=dataset_name,
            corruption='zoom_blur',
            split='test',
            severity=5
        )
        
        print(f"Clean dataset loaded with {len(clean_dataset)} samples")
        
        # Generate and display a montage
        montage = clean_dataset.montage(length=5)
        plt.figure(figsize=(10, 10))
        plt.imshow(montage)
        plt.title(f"{dataset_name} - Clean")
        plt.axis('off')
        plt.savefig(f"montage_{dataset_name}_clean.jpg")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check if the dataset exists at the specified path.") 