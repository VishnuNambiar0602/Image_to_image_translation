"""
CRN: Cascaded Refinement Networks for Image Generation
Feed-forward refinement achieving FID: 41.8 (Fastest training & inference)
"""

from .models import RefinementBlock, RefinementStage, CRNGenerator, CRNWithMultiScale, CRN
from .config import CRNConfig

__version__ = "1.0.0"
__all__ = [
    "RefinementBlock",
    "RefinementStage",
    "CRNGenerator",
    "CRNWithMultiScale",
    "CRN",
    "CRNConfig"
]
