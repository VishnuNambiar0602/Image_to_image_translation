"""
Utility functions for Pix2Pix training and inference.
Includes logging, visualization, checkpoint management, etc.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import logging
from PIL import Image
from torchvision.utils import make_grid, save_image
import os


def setup_logging(log_dir: str, name: str = 'pix2pix') -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        log_dir: Directory to save logs
        name: Logger name
        
    Returns:
        Logger object
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(name)
    return logger


class CheckpointManager:
    """Manage model checkpoints."""
    
    def __init__(self, checkpoint_dir: str, keep_best: int = 3):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_best: Number of best checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_best = keep_best
        self.best_metrics = []
    
    def save_checkpoint(self, generator: nn.Module, discriminator: nn.Module,
                       optimizer_g: torch.optim.Optimizer,
                       optimizer_d: torch.optim.Optimizer,
                       epoch: int, metrics: Dict,
                       is_best: bool = False):
        """Save a checkpoint.
        
        Args:
            generator: Generator model
            discriminator: Discriminator model
            optimizer_g: Generator optimizer
            optimizer_d: Discriminator optimizer
            epoch: Current epoch
            metrics: Metrics dictionary
            is_best: Whether this is the best checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'optimizer_g': optimizer_g.state_dict(),
            'optimizer_d': optimizer_d.state_dict(),
            'metrics': metrics
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'checkpoint_best.pt'
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path: str, generator: nn.Module,
                       discriminator: nn.Module,
                       optimizer_g: Optional[torch.optim.Optimizer] = None,
                       optimizer_d: Optional[torch.optim.Optimizer] = None):
        """Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            generator: Generator model
            discriminator: Discriminator model
            optimizer_g: Generator optimizer (optional)
            optimizer_d: Discriminator optimizer (optional)
            
        Returns:
            Dictionary with checkpoint info
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        generator.load_state_dict(checkpoint['generator'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        
        if optimizer_g is not None:
            optimizer_g.load_state_dict(checkpoint['optimizer_g'])
        
        if optimizer_d is not None:
            optimizer_d.load_state_dict(checkpoint['optimizer_d'])
        
        return {
            'epoch': checkpoint.get('epoch', 0),
            'metrics': checkpoint.get('metrics', {})
        }


class ImageBuffer:
    """Implements an image buffer that stores previously generated images.
    
    This buffer allows the discriminator to be updated with images generated
    previously, rather than just the most recent batch. This helps stabilize
    GAN training.
    """
    
    def __init__(self, pool_size: int = 50):
        """
        Args:
            pool_size: Maximum number of images to store
        """
        self.pool_size = pool_size
        self.num_imgs = 0
        self.images = []
    
    def query(self, images: torch.Tensor) -> torch.Tensor:
        """Query and update buffer.
        
        Args:
            images: Current batch of generated images
            
        Returns:
            Batch of images (mix of current and previously generated)
        """
        if self.pool_size == 0:
            return images
        
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            
            if self.num_imgs < self.pool_size:
                self.num_imgs += 1
                self.images.append(image)
                return_images.append(image)
            else:
                if np.random.uniform(0, 1) > 0.5:
                    random_id = np.random.randint(0, self.pool_size)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        
        return_images = torch.cat(return_images, 0)
        return return_images


def denormalize_image(image: torch.Tensor) -> torch.Tensor:
    """Denormalize image from [-1, 1] to [0, 1].
    
    Args:
        image: Normalized image tensor
        
    Returns:
        Denormalized image tensor
    """
    return (image + 1) / 2


def normalize_image(image: torch.Tensor) -> torch.Tensor:
    """Normalize image from [0, 1] to [-1, 1].
    
    Args:
        image: Image tensor in [0, 1] range
        
    Returns:
        Normalized image tensor
    """
    return image * 2 - 1


def visualize_batch(source: torch.Tensor, generated: torch.Tensor,
                   target: Optional[torch.Tensor] = None,
                   save_path: Optional[str] = None,
                   num_images: int = 4):
    """Visualize a batch of images side by side.
    
    Args:
        source: Source image tensor [B, C, H, W]
        generated: Generated image tensor [B, C, H, W]
        target: Target image tensor [B, C, H, W] (optional)
        save_path: Path to save visualization
        num_images: Number of images to visualize
    """
    # Select first num_images
    source = source[:num_images]
    generated = generated[:num_images]
    if target is not None:
        target = target[:num_images]
    
    # Denormalize
    source = denormalize_image(source)
    generated = denormalize_image(generated)
    if target is not None:
        target = denormalize_image(target)
    
    # Concatenate images
    if target is not None:
        images = torch.cat([source, generated, target], dim=0)
        grid = make_grid(images, nrow=num_images, normalize=False,
                        value_range=(0, 1), padding=2)
    else:
        images = torch.cat([source, generated], dim=0)
        grid = make_grid(images, nrow=num_images, normalize=False,
                        value_range=(0, 1), padding=2)
    
    # Save or display
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_image(grid, save_path)
    
    return grid


def plot_losses(losses_g: List[float], losses_d: List[float],
               save_path: Optional[str] = None):
    """Plot generator and discriminator losses.
    
    Args:
        losses_g: Generator loss values
        losses_d: Discriminator loss values
        save_path: Path to save plot
    """
    plt.figure(figsize=(10, 5))
    
    plt.plot(losses_g, label='Generator Loss', alpha=0.7)
    plt.plot(losses_d, label='Discriminator Loss', alpha=0.7)
    
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Pix2Pix Training Losses')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    plt.close()


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed: int):
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_image(image_path: str, size: Tuple[int, int] = (256, 256)) -> torch.Tensor:
    """Load and preprocess an image.
    
    Args:
        image_path: Path to image file
        size: Target image size
        
    Returns:
        Image tensor normalized to [-1, 1]
    """
    from torchvision import transforms
    
    image = Image.open(image_path).convert('RGB')
    image = image.resize(size, Image.BICUBIC)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    return transform(image).unsqueeze(0)


def save_results(results: Dict, save_path: str):
    """Save results dictionary to JSON file.
    
    Args:
        results: Results dictionary
        save_path: Path to save JSON file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    results_serializable = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            results_serializable[key] = value.tolist()
        elif isinstance(value, (np.floating, np.integer)):
            results_serializable[key] = float(value)
        else:
            results_serializable[key] = value
    
    with open(save_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)


def get_device() -> str:
    """Get the best available device.
    
    Returns:
        'cuda' if GPU available, else 'cpu'
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def get_torch_version() -> str:
    """Get PyTorch version."""
    return torch.__version__


# Export
__all__ = ['setup_logging', 'CheckpointManager', 'ImageBuffer',
           'denormalize_image', 'normalize_image', 'visualize_batch',
           'plot_losses', 'count_parameters', 'set_seed', 'load_image',
           'save_results', 'get_device', 'get_torch_version']
