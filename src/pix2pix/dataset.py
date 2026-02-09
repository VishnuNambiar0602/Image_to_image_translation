"""
Dataset classes and utilities for Pix2Pix training.
Supports loading paired image datasets for image-to-image translation.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
from PIL import Image
import os
from typing import Tuple, Optional, Callable, List
from config import Config


class PairedImageDataset(Dataset):
    """Dataset for paired image translation.
    
    Loads paired source and target images from directories.
    Images should be organized as:
    - dataset_path/
        - train/
            - source/
            - target/
        - test/
            - source/
            - target/
    """
    
    def __init__(self, root_dir: str, split: str = 'train', 
                 image_size: Tuple[int, int] = (256, 256),
                 augment: bool = True, file_extension: str = '.jpg'):
        """
        Args:
            root_dir: Root directory of the dataset
            split: 'train' or 'test'
            image_size: Size to resize images to
            augment: Whether to apply data augmentation
            file_extension: Extension of image files
        """
        
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.file_extension = file_extension
        
        # Paths to source and target directories
        self.source_dir = self.root_dir / split / 'source'
        self.target_dir = self.root_dir / split / 'target'
        
        # Get list of image filenames
        self.image_filenames = sorted([
            f for f in os.listdir(self.source_dir) 
            if f.endswith(file_extension)
        ])
        
        # Data augmentation pipeline
        self.augmentation_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]) if augment else None
        
        # Normalization
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                               std=[0.5, 0.5, 0.5])
        ])
    
    def __len__(self) -> int:
        return len(self.image_filenames)
    
    def __getitem__(self, idx: int) -> dict:
        filename = self.image_filenames[idx]
        
        # Load images
        source_path = self.source_dir / filename
        target_path = self.target_dir / filename
        
        source_img = Image.open(source_path).convert('RGB')
        target_img = Image.open(target_path).convert('RGB')
        
        # Resize
        source_img = source_img.resize(self.image_size, Image.BICUBIC)
        target_img = target_img.resize(self.image_size, Image.BICUBIC)
        
        # Apply augmentation
        if self.augmentation_transforms is not None:
            # Apply same augmentation to both images
            seed = np.random.randint(0, 2147483647)
            
            np.random.seed(seed)
            torch.manual_seed(seed)
            source_img = self.augmentation_transforms(source_img)
            
            np.random.seed(seed)
            torch.manual_seed(seed)
            target_img = self.augmentation_transforms(target_img)
        
        # Normalize
        source_tensor = self.normalize(source_img)
        target_tensor = self.normalize(target_img)
        
        return {
            'source': source_tensor,
            'target': target_tensor,
            'filename': filename
        }


class SingleImageDataset(Dataset):
    """Dataset for inference on single images.
    
    Useful for testing on individual images without paired targets.
    """
    
    def __init__(self, image_dir: str, 
                 image_size: Tuple[int, int] = (256, 256),
                 file_extension: str = '.jpg'):
        """
        Args:
            image_dir: Directory containing images
            image_size: Size to resize images to
            file_extension: Extension of image files
        """
        
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        
        # Get list of image filenames
        self.image_filenames = sorted([
            f for f in os.listdir(self.image_dir) 
            if f.endswith(file_extension)
        ])
        
        # Normalization
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                               std=[0.5, 0.5, 0.5])
        ])
    
    def __len__(self) -> int:
        return len(self.image_filenames)
    
    def __getitem__(self, idx: int) -> dict:
        filename = self.image_filenames[idx]
        image_path = self.image_dir / filename
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Resize
        image = image.resize(self.image_size, Image.BICUBIC)
        
        # Normalize
        image_tensor = self.normalize(image)
        
        return {
            'image': image_tensor,
            'filename': filename
        }


class DatasetFactory:
    """Factory for creating datasets."""
    
    @staticmethod
    def create_dataloader(dataset_path: str,
                         split: str = 'train',
                         batch_size: int = 1,
                         num_workers: int = 4,
                         shuffle: bool = True,
                         image_size: Tuple[int, int] = (256, 256),
                         augment: bool = True) -> DataLoader:
        """Create a DataLoader for paired image translation.
        
        Args:
            dataset_path: Path to dataset directory
            split: 'train' or 'test'
            batch_size: Batch size
            num_workers: Number of workers for data loading
            shuffle: Whether to shuffle data
            image_size: Image size to resize to
            augment: Whether to apply augmentation
            
        Returns:
            DataLoader object
        """
        
        dataset = PairedImageDataset(
            root_dir=dataset_path,
            split=split,
            image_size=image_size,
            augment=augment and split == 'train'
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle and split == 'train',
            num_workers=num_workers,
            pin_memory=True,
            drop_last=split == 'train'
        )
        
        return dataloader
    
    @staticmethod
    def create_inference_dataloader(image_dir: str,
                                   batch_size: int = 8,
                                   num_workers: int = 4,
                                   image_size: Tuple[int, int] = (256, 256)) -> DataLoader:
        """Create a DataLoader for inference.
        
        Args:
            image_dir: Path to directory containing images
            batch_size: Batch size
            num_workers: Number of workers for data loading
            image_size: Image size to resize to
            
        Returns:
            DataLoader object
        """
        
        dataset = SingleImageDataset(
            image_dir=image_dir,
            image_size=image_size
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return dataloader


# Export
__all__ = ['PairedImageDataset', 'SingleImageDataset', 'DatasetFactory']
