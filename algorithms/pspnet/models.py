"""
PSPNet: Pyramid Scene Parsing Network for Semantic Segmentation
Traditional segmentation + photorealism enhancement approach.

PSPNet uses pyramid pooling to capture multi-scale contextual information
for semantic understanding, then applies post-processing for photorealism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class ConvBlock(nn.Module):
    """Convolutional block with batch normalization and activation."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, dilation: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class PyramidPooling(nn.Module):
    """Pyramid pooling module for multi-scale feature extraction."""
    
    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        self.pool_sizes = [1, 2, 3, 6]
        mid_channels = in_channels // reduction
        
        self.pools = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                ConvBlock(in_channels, mid_channels, kernel_size=1, padding=0)
            )
            for pool_size in self.pool_sizes
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.size()[2:]
        pool_outputs = [x]
        
        for pool in self.pools:
            pooled = pool(x)
            upsampled = F.interpolate(pooled, size=(h, w), mode='bilinear', align_corners=False)
            pool_outputs.append(upsampled)
        
        return torch.cat(pool_outputs, dim=1)


class PSPNetSegmentation(nn.Module):
    """PSPNet for semantic segmentation.
    
    Pyramid Scene Parsing Network that captures multi-scale contextual
    information for accurate semantic segmentation.
    """
    
    def __init__(self, num_classes: int = 150, in_channels: int = 3, features: int = 512):
        super().__init__()
        
        # Initial convolution layers
        self.initial = nn.Sequential(
            ConvBlock(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # ResNet-style encoder
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, features, 3, stride=2)
        
        # Pyramid pooling
        self.pyramid_pool = PyramidPooling(features)
        
        # Segmentation head
        pool_channels = features + (features // 4) * 4
        self.seg_head = nn.Sequential(
            ConvBlock(pool_channels, 128, kernel_size=3, padding=1),
            nn.Dropout2d(0.1),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )
    
    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int = 1):
        layers = []
        layers.append(ConvBlock(in_channels, out_channels, stride=stride))
        for _ in range(1, blocks):
            layers.append(ConvBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.size()[2:]
        
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.pyramid_pool(x)
        x = self.seg_head(x)
        
        # Upsample to original resolution
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        return x


class PhotorealisticEnhancer(nn.Module):
    """Post-processing module to enhance photorealism of segmentation outputs."""
    
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.enhance = nn.Sequential(
            ConvBlock(in_channels, 64, kernel_size=3, padding=1),
            ConvBlock(64, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.size()[2:]
        residual = x
        x = self.enhance(x)
        return x * residual + (1 - x) * 0.5


class PSPNet(nn.Module):
    """Complete PSPNet with photorealism enhancement."""
    
    def __init__(self, num_classes: int = 150):
        super().__init__()
        self.segmentation = PSPNetSegmentation(num_classes)
        self.enhancer = PhotorealisticEnhancer(3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seg_output = self.segmentation(x)
        # Convert segmentation to RGB for enhancement
        segmentation_rgb = torch.softmax(seg_output, dim=1).mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        enhanced = self.enhancer(segmentation_rgb * x)
        return enhanced


__all__ = [
    "ConvBlock",
    "PyramidPooling",
    "PSPNetSegmentation",
    "PhotorealisticEnhancer",
    "PSPNet"
]
