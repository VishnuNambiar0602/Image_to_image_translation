"""
CycleGAN: Unpaired Image-to-Image Translation
Architecture: Dual generators and discriminators with cycle-consistency loss.

CycleGAN learns to translate images from domain X to domain Y without paired examples.
It uses two generator-discriminator pairs with cycle-consistency loss to preserve semantics.
"""

import torch
import torch.nn as nn
from typing import Tuple


class ResidualBlock(nn.Module):
    """Residual block for CycleGAN generator."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0),
            nn.InstanceNorm2d(out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class CycleGANGenerator(nn.Module):
    """CycleGAN Generator with residual blocks.
    
    Learns mappings between unpaired image domains.
    Uses residual blocks for stable training and semantic preservation.
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3, 
                 features: int = 64, num_residuals: int = 9):
        super().__init__()
        
        # Initial convolution
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, features, kernel_size=7, padding=0),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True)
        )
        
        # Downsampling
        self.down1 = nn.Sequential(
            nn.Conv2d(features, features*2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(features*2),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(features*2, features*4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(features*4),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        self.residuals = nn.Sequential(*[
            ResidualBlock(features*4, features*4) for _ in range(num_residuals)
        ])
        
        # Upsampling
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(features*4, features*2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(features*2),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(features*2, features, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True)
        )
        
        # Output layer
        self.output = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(features, out_channels, kernel_size=7, padding=0),
            nn.Tanh()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.residuals(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.output(x)
        return x


class CycleGANDiscriminator(nn.Module):
    """CycleGAN Discriminator using PatchGAN architecture."""
    
    def __init__(self, in_channels: int = 3, features: int = 64):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(features, features*2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(features*2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(features*2, features*4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(features*4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(features*4, features*8, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(features*8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(features*8, 1, kernel_size=4, stride=1, padding=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class CycleGAN(nn.Module):
    """Complete CycleGAN model with dual generators and discriminators."""
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()
        
        # Generators
        self.generator_x2y = CycleGANGenerator(in_channels, out_channels)
        self.generator_y2x = CycleGANGenerator(in_channels, out_channels)
        
        # Discriminators
        self.discriminator_x = CycleGANDiscriminator(in_channels)
        self.discriminator_y = CycleGANDiscriminator(in_channels)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for both domains.
        
        Args:
            x: Images from domain X
            y: Images from domain Y
            
        Returns:
            Fake Y, Cycle X, Fake X, Cycle Y
        """
        fake_y = self.generator_x2y(x)
        cycle_x = self.generator_y2x(fake_y)
        
        fake_x = self.generator_y2x(y)
        cycle_y = self.generator_x2y(fake_x)
        
        return fake_y, cycle_x, fake_x, cycle_y


__all__ = [
    "ResidualBlock",
    "CycleGANGenerator",
    "CycleGANDiscriminator",
    "CycleGAN"
]
