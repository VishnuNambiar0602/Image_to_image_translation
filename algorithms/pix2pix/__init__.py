"""
Pix2Pix: Image-to-Image Translation with Conditional GANs (Primary Algorithm)
Paired image translation achieving FID: 26.3 (BEST QUALITY)
"""

__version__ = "2.0.0"

try:
    # Import from src if available in main project
    from src.pix2pix import models, config
except ImportError:
    # Fallback to local config
    from . import config

__all__ = ["config"]
