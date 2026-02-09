"""
Pix2Pix: Image-to-Image Translation with Conditional GANs

A comprehensive PyTorch implementation for paired image domain translation.
"""

__version__ = "1.0.0"
__author__ = "GAN_CIA Research"
__date__ = "February 2026"

from .config import Config
from .models import UNetGenerator, PatchGANDiscriminator, Pix2PixModel
from .losses import Pix2PixLoss, GANLoss, L1Loss
from .dataset import PairedImageDataset, DatasetFactory
from .metrics import MetricComputer, FrechetInceptionDistance, InceptionScore
from .utils import CheckpointManager, ImageBuffer, set_seed

__all__ = [
    'Config',
    'UNetGenerator',
    'PatchGANDiscriminator',
    'Pix2PixModel',
    'Pix2PixLoss',
    'GANLoss',
    'L1Loss',
    'PairedImageDataset',
    'DatasetFactory',
    'MetricComputer',
    'FrechetInceptionDistance',
    'InceptionScore',
    'CheckpointManager',
    'ImageBuffer',
    'set_seed',
]

__doc__ = """
Pix2Pix Implementation

This package provides a complete implementation of Pix2Pix for image-to-image translation.

Key Components:
- Models: U-Net Generator + PatchGAN Discriminator
- Losses: Adversarial + L1 reconstruction loss
- Datasets: Support for multiple paired image datasets
- Metrics: FID, IS, LPIPS, SSIM, PSNR evaluation
- Training: Full training pipeline with checkpointing
- Inference: Single and batch inference

Example Usage:
    from pix2pix import UNetGenerator, PatchGANDiscriminator, Config
    
    generator = UNetGenerator().cuda()
    discriminator = PatchGANDiscriminator().cuda()
    
    config = Config()
    # Use for training and inference
"""
