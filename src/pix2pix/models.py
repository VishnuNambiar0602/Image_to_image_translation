"""
Generator and Discriminator architectures for Pix2Pix.
Includes U-Net Generator and PatchGAN Discriminator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class ConvBlock(nn.Module):
    """Convolution block with normalization and activation."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 4,
                 stride: int = 1, padding: int = 1, norm_type: str = 'batch',
                 activation: str = 'relu', use_bias: bool = True):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                              stride, padding, bias=use_bias)
        
        # Normalization
        if norm_type == 'batch':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm_type == 'instance':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm_type == 'layer':
            self.norm = nn.GroupNorm(1, out_channels)
        else:
            self.norm = nn.Identity()
        
        # Activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class SpectralNormConv2d(nn.Module):
    """Spectral Normalized Convolution."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, bias: bool = True):
        super().__init__()
        self.conv = nn.utils.spectral_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNetGenerator(nn.Module):
    """U-Net Generator for Pix2Pix.
    
    Encoder-Decoder architecture with skip connections.
    Input: source image (3 channels)
    Output: target domain image (3 channels)
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3,
                 features: int = 64, use_skip: bool = True,
                 norm_type: str = 'instance'):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.use_skip = use_skip
        
        # Encoder (downsampling)
        self.enc1 = ConvBlock(in_channels, features, kernel_size=4, stride=2, 
                             padding=1, norm_type='none', activation='leaky_relu')
        self.enc2 = ConvBlock(features, features*2, kernel_size=4, stride=2, 
                             padding=1, norm_type=norm_type, activation='leaky_relu')
        self.enc3 = ConvBlock(features*2, features*4, kernel_size=4, stride=2, 
                             padding=1, norm_type=norm_type, activation='leaky_relu')
        self.enc4 = ConvBlock(features*4, features*8, kernel_size=4, stride=2, 
                             padding=1, norm_type=norm_type, activation='leaky_relu')
        self.enc5 = ConvBlock(features*8, features*8, kernel_size=4, stride=2, 
                             padding=1, norm_type=norm_type, activation='leaky_relu')
        self.enc6 = ConvBlock(features*8, features*8, kernel_size=4, stride=2, 
                             padding=1, norm_type=norm_type, activation='leaky_relu')
        self.enc7 = ConvBlock(features*8, features*8, kernel_size=4, stride=2, 
                             padding=1, norm_type=norm_type, activation='leaky_relu')
        
        # Bottleneck
        self.bottleneck = ConvBlock(features*8, features*8, kernel_size=4, stride=2, 
                                   padding=1, norm_type=norm_type, activation='leaky_relu')
        
        # Decoder (upsampling)
        self.dec1 = self._deconv_block(features*8, features*8, norm_type=norm_type)
        self.dec2 = self._deconv_block(features*16 if use_skip else features*8, 
                                      features*8, norm_type=norm_type)
        self.dec3 = self._deconv_block(features*16 if use_skip else features*8, 
                                      features*8, norm_type=norm_type)
        self.dec4 = self._deconv_block(features*16 if use_skip else features*8, 
                                      features*8, norm_type=norm_type)
        self.dec5 = self._deconv_block(features*16 if use_skip else features*8, 
                                      features*4, norm_type=norm_type)
        self.dec6 = self._deconv_block(features*8 if use_skip else features*4, 
                                      features*2, norm_type=norm_type)
        self.dec7 = self._deconv_block(features*4 if use_skip else features*2, 
                                      features, norm_type=norm_type)
        
        # Output layer
        self.output = nn.Sequential(
            nn.ConvTranspose2d(features*2 if use_skip else features, out_channels, 
                             kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def _deconv_block(self, in_channels: int, out_channels: int, 
                     norm_type: str = 'instance') -> nn.Module:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, 
                             stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels) if norm_type == 'batch' else nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        
        # Bottleneck
        b = self.bottleneck(e7)
        
        # Decoder with skip connections
        d1 = self.dec1(b)
        d1 = torch.cat([d1, e7], dim=1) if self.use_skip else d1
        
        d2 = self.dec2(d1)
        d2 = torch.cat([d2, e6], dim=1) if self.use_skip else d2
        
        d3 = self.dec3(d2)
        d3 = torch.cat([d3, e5], dim=1) if self.use_skip else d3
        
        d4 = self.dec4(d3)
        d4 = torch.cat([d4, e4], dim=1) if self.use_skip else d4
        
        d5 = self.dec5(d4)
        d5 = torch.cat([d5, e3], dim=1) if self.use_skip else d5
        
        d6 = self.dec6(d5)
        d6 = torch.cat([d6, e2], dim=1) if self.use_skip else d6
        
        d7 = self.dec7(d6)
        d7 = torch.cat([d7, e1], dim=1) if self.use_skip else d7
        
        # Output
        output = self.output(d7)
        
        return output


class PatchGANDiscriminator(nn.Module):
    """PatchGAN Discriminator for Pix2Pix.
    
    Classifies overlapping patches of an image as real or fake.
    Effective for high-frequency details and local structure.
    
    Input: concatenated source and target images (6 channels)
    Output: Patch-wise classification (70x70 receptive field)
    """
    
    def __init__(self, in_channels: int = 6, features: int = 64,
                 use_spectral_norm: bool = True, use_sigmoid: bool = True):
        super().__init__()
        
        self.in_channels = in_channels
        self.features = features
        self.use_spectral_norm = use_spectral_norm
        
        # Build discriminator blocks
        if use_spectral_norm:
            conv_fn = SpectralNormConv2d
        else:
            conv_fn = nn.Conv2d
        
        # Block 1: input -> 64
        self.block1 = nn.Sequential(
            conv_fn(in_channels, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Block 2: 64 -> 128
        self.block2 = nn.Sequential(
            conv_fn(features, features*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(features*2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Block 3: 128 -> 256
        self.block3 = nn.Sequential(
            conv_fn(features*2, features*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(features*4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Block 4: 256 -> 512
        self.block4 = nn.Sequential(
            conv_fn(features*4, features*8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(features*8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Output layer: 512 -> 1
        self.output = conv_fn(features*8, 1, kernel_size=4, stride=1, padding=1)
        
        if use_sigmoid:
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, 6, H, W) - concatenated source and target
            
        Returns:
            Classification output of shape (B, 1, H//16, W//16) - patch-wise predictions
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.output(x)
        
        if self.sigmoid is not None:
            x = self.sigmoid(x)
        
        return x


class Pix2PixModel(nn.Module):
    """Complete Pix2Pix model combining generator and discriminator."""
    
    def __init__(self, config):
        super().__init__()
        
        self.generator = UNetGenerator(
            in_channels=config.model.GENERATOR_IN_CHANNELS,
            out_channels=config.model.GENERATOR_OUT_CHANNELS,
            features=config.model.GENERATOR_FEATURES,
            use_skip=config.model.SKIP_CONNECTIONS,
            norm_type=config.model.GENERATOR_NORM_TYPE
        )
        
        self.discriminator = PatchGANDiscriminator(
            in_channels=config.model.DISCRIMINATOR_IN_CHANNELS,
            features=config.model.DISCRIMINATOR_FEATURES,
            use_spectral_norm=config.model.USE_SPECTRAL_NORM
        )
    
    def forward(self, source: torch.Tensor) -> torch.Tensor:
        return self.generator(source)


# Export
__all__ = ['ConvBlock', 'SpectralNormConv2d', 'UNetGenerator', 
           'PatchGANDiscriminator', 'Pix2PixModel']
