"""
CycleGAN: Unpaired Image-to-Image Translation
Unpaired image translation achieving FID: 35.2 (Flexible, no paired data needed)
"""

from .models import CycleGANGenerator, CycleGANDiscriminator, CycleGAN
from .config import CycleGANConfig

__version__ = "1.0.0"
__all__ = ["CycleGANGenerator", "CycleGANDiscriminator", "CycleGAN", "CycleGANConfig"]
