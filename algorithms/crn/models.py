"""
CRN: Cascaded Refinement Networks for Image Generation
Feed-forward refinement architecture without adversarial training.

CRN uses cascaded refinement stages to progressively improve image quality
while being faster to train than adversarial approaches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RefinementBlock(nn.Module):
    """Single refinement block for cascaded refinement."""
    
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x + residual


class RefinementStage(nn.Module):
    """Single refinement stage with multiple refinement blocks."""
    
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 3):
        super().__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.refinement_blocks = nn.Sequential(*[
            RefinementBlock(out_channels) for _ in range(num_blocks)
        ])
        
        self.output_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial_conv(x)
        x = self.refinement_blocks(x)
        x = self.output_conv(x)
        return x


class CRNGenerator(nn.Module):
    """CRN Generator with cascaded refinement stages.
    
    Uses multiple refinement stages to progressively improve image quality
    in a feed-forward manner without adversarial training.
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3, 
                 num_stages: int = 4, base_channels: int = 64):
        super().__init__()
        
        self.num_stages = num_stages
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Cascaded refinement stages
        self.stages = nn.ModuleList([
            RefinementStage(base_channels, base_channels, num_blocks=3)
            for _ in range(num_stages)
        ])
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode
        features = self.encoder(x)
        
        # Cascaded refinement
        for stage in self.stages:
            features = stage(features)
        
        # Decode
        output = self.decoder(features)
        return output


class CRNWithMultiScale(nn.Module):
    """CRN with multi-scale refinement for improved quality."""
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3, num_scales: int = 3):
        super().__init__()
        self.num_scales = num_scales
        
        # Multi-scale CRN generators
        self.generators = nn.ModuleList([
            CRNGenerator(in_channels, out_channels, num_stages=4, base_channels=64)
            for _ in range(num_scales)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        current_input = x
        
        for i, generator in enumerate(self.generators):
            output = generator(current_input)
            outputs.append(output)
            
            # For next scale, downsample (except for last scale)
            if i < self.num_scales - 1:
                current_input = F.adaptive_avg_pool2d(output, (x.size(2)//2, x.size(3)//2))
        
        # Use final output
        return outputs[-1]


class CRN(nn.Module):
    """Complete CRN model combining multiple refinement stages."""
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()
        self.generator = CRNWithMultiScale(in_channels, out_channels, num_scales=3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.generator(x)


__all__ = [
    "RefinementBlock",
    "RefinementStage",
    "CRNGenerator",
    "CRNWithMultiScale",
    "CRN"
]
