"""
Unit tests for dataset loading and preprocessing.
"""

import pytest
import torch
from pathlib import Path
from src.pix2pix.dataset import PairedImageDataset
from src.pix2pix.config import Config


class TestPairedImageDataset:
    """Test suite for PairedImageDataset."""

    @pytest.mark.unit
    def test_dataset_initialization(self):
        """Test dataset can be initialized."""
        dataset_path = Path("datasets/cityscapes")
        if dataset_path.exists():
            dataset = PairedImageDataset(
                dataset_path / "train",
                image_size=256,
                augment=False
            )
            assert len(dataset) > 0

    @pytest.mark.unit
    def test_dataset_sample_shape(self):
        """Test dataset returns correct tensor shapes."""
        dataset_path = Path("datasets/cityscapes")
        if dataset_path.exists():
            dataset = PairedImageDataset(
                dataset_path / "train",
                image_size=256,
                augment=False
            )
            sample = dataset[0]

            assert "source" in sample
            assert "target" in sample
            assert sample["source"].shape == (3, 256, 256)
            assert sample["target"].shape == (3, 256, 256)

    @pytest.mark.unit
    def test_dataset_value_range(self):
        """Test dataset normalizes values to [-1, 1]."""
        dataset_path = Path("datasets/cityscapes")
        if dataset_path.exists():
            dataset = PairedImageDataset(
                dataset_path / "train",
                image_size=256,
                augment=False
            )
            sample = dataset[0]

            assert sample["source"].max() <= 1.0
            assert sample["source"].min() >= -1.0
            assert sample["target"].max() <= 1.0
            assert sample["target"].min() >= -1.0


@pytest.mark.unit
def test_dataloader():
    """Test dataloader creation and batching."""
    dataset_path = Path("datasets/cityscapes")
    if dataset_path.exists():
        dataset = PairedImageDataset(
            dataset_path / "train",
            image_size=256,
            augment=False
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=2,
            shuffle=False
        )

        batch = next(iter(dataloader))
        assert batch["source"].shape == (2, 3, 256, 256)
        assert batch["target"].shape == (2, 3, 256, 256)
