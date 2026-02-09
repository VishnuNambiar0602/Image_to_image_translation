"""
Inference script for Pix2Pix model.
Generate domain translations on new images.
"""

import torch
import torch.nn as nn
from pathlib import Path
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm

from config import Config
from models import UNetGenerator
from dataset import DatasetFactory
from utils import load_image, denormalize_image, get_device, logger
from torchvision.utils import save_image


class Pix2PixInference:
    """Inference class for Pix2Pix model."""
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        """
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on
        """
        self.device = device
        
        # Load generator
        self.generator = UNetGenerator(
            in_channels=Config.model.GENERATOR_IN_CHANNELS,
            out_channels=Config.model.GENERATOR_OUT_CHANNELS,
            features=Config.model.GENERATOR_FEATURES,
            use_skip=Config.model.SKIP_CONNECTIONS,
            norm_type=Config.model.GENERATOR_NORM_TYPE
        ).to(device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'generator' in checkpoint:
            self.generator.load_state_dict(checkpoint['generator'])
        else:
            self.generator.load_state_dict(checkpoint)
        
        self.generator.eval()
        print(f"Model loaded from {checkpoint_path}")
    
    def translate_image(self, input_image: torch.Tensor) -> torch.Tensor:
        """Translate a single image.
        
        Args:
            input_image: Input image tensor [1, C, H, W] or [C, H, W]
            
        Returns:
            Generated image tensor
        """
        if input_image.dim() == 3:
            input_image = input_image.unsqueeze(0)
        
        input_image = input_image.to(self.device)
        
        with torch.no_grad():
            generated = self.generator(input_image)
        
        return generated.cpu()
    
    def translate_batch(self, image_dir: str, output_dir: str,
                       image_size: tuple = (256, 256)):
        """Translate a batch of images.
        
        Args:
            image_dir: Directory containing input images
            output_dir: Directory to save output images
            image_size: Size to resize images to
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dataloader
        val_loader = DatasetFactory.create_inference_dataloader(
            image_dir=image_dir,
            batch_size=Config.inference.BATCH_SIZE,
            image_size=image_size
        )
        
        # Process batch
        for batch_idx, batch in enumerate(tqdm(val_loader)):
            images = batch['image']
            filenames = batch['filename']
            
            # Generate
            generated = self.translate_image(images)
            
            # Save outputs
            for img, filename in zip(generated, filenames):
                output_path = output_dir / f'generated_{filename}'
                
                # Denormalize and save
                img_viz = denormalize_image(img.unsqueeze(0))
                save_image(img_viz, output_path)
    
    def visualize_translation(self, source_path: str, output_path: str = None,
                             image_size: tuple = (256, 256)):
        """Visualize input and output side by side.
        
        Args:
            source_path: Path to source image
            output_path: Path to save visualization
            image_size: Size to resize images to
        """
        # Load image
        source = load_image(source_path, image_size).to(self.device)
        
        # Generate
        generated = self.translate_image(source)
        
        # Denormalize
        source_viz = denormalize_image(source)
        generated_viz = denormalize_image(generated)
        
        # Create visualization
        import torch.nn.functional as F
        from torchvision.utils import make_grid
        
        viz = torch.cat([source_viz, generated_viz], dim=0)
        grid = make_grid(viz, nrow=1, normalize=False, value_range=(0, 1))
        
        if output_path:
            save_image(grid, output_path)
        
        return grid


def main():
    parser = argparse.ArgumentParser(description='Pix2Pix Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--input-dir', type=str,
                       help='Directory with input images')
    parser.add_argument('--input-image', type=str,
                       help='Single input image path')
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='Output directory')
    parser.add_argument('--image-size', type=int, nargs=2, default=[256, 256],
                       help='Image size')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda or cpu)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for batch inference')
    
    args = parser.parse_args()
    
    # Get device
    device = args.device or get_device()
    
    # Create inferencer
    inferencer = Pix2PixInference(args.checkpoint, device=device)
    
    # Inference
    if args.input_image:
        # Single image
        print(f"Translating {args.input_image}...")
        output_path = Path(args.output_dir) / 'generated.png'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        inferencer.visualize_translation(
            args.input_image,
            output_path,
            image_size=tuple(args.image_size)
        )
        print(f"Output saved to {output_path}")
    
    elif args.input_dir:
        # Batch inference
        print(f"Translating images from {args.input_dir}...")
        inferencer.translate_batch(
            args.input_dir,
            args.output_dir,
            image_size=tuple(args.image_size)
        )
        print(f"Results saved to {args.output_dir}")
    
    else:
        print("Error: Specify either --input-image or --input-dir")


if __name__ == '__main__':
    main()
