"""
Conftest for pytest configuration.
Shared fixtures and configurations for all tests.
"""

import pytest
import torch


@pytest.fixture
def device():
    """Fixture to provide device (GPU if available, else CPU)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def sample_image():
    """Fixture to provide a sample image tensor."""
    return torch.randn(1, 3, 256, 256)


@pytest.fixture
def batch_images():
    """Fixture to provide a batch of image tensors."""
    return torch.randn(4, 3, 256, 256)
