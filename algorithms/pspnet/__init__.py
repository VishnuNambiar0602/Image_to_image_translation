"""
PSPNet: Pyramid Scene Parsing Network for Semantic Segmentation
Traditional segmentation achieving FID: 47.2 (Interpretable but lower quality)
"""

from .models import PSPNetSegmentation, PhotorealisticEnhancer, PSPNet
from .config import PSPNetConfig

__version__ = "1.0.0"
__all__ = ["PSPNetSegmentation", "PhotorealisticEnhancer", "PSPNet", "PSPNetConfig"]
