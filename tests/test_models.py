"""
Unit tests for Pix2Pix models.
Tests generator, discriminator, and model architectures.
"""

import pytest
import torch
from src.pix2pix.models import UNetGenerator, PatchGANDiscriminator, Pix2PixModel
from src.pix2pix.config import Config


class TestUNetGenerator:
    """Test suite for U-Net Generator."""

    @pytest.mark.unit
    def test_generator_output_shape(self):
        """Test generator output shape matches input shape."""
        generator = UNetGenerator(in_channels=3, out_channels=3, features=64)
        batch_size, channels, height, width = 2, 3, 256, 256
        input_tensor = torch.randn(batch_size, channels, height, width)

        output = generator(input_tensor)

        assert output.shape == input_tensor.shape
        assert output.dtype == torch.float32

    @pytest.mark.unit
    def test_generator_different_sizes(self):
        """Test generator with different input sizes."""
        generator = UNetGenerator()
        sizes = [128, 256, 512]

        for size in sizes:
            input_tensor = torch.randn(1, 3, size, size)
            output = generator(input_tensor)
            assert output.shape == (1, 3, size, size)

    @pytest.mark.unit
    def test_generator_output_range(self):
        """Test generator output is in [-1, 1] range (Tanh activation)."""
        generator = UNetGenerator()
        input_tensor = torch.randn(1, 3, 256, 256)

        output = generator(input_tensor)

        assert output.max() <= 1.0
        assert output.min() >= -1.0


class TestPatchGANDiscriminator:
    """Test suite for PatchGAN Discriminator."""

    @pytest.mark.unit
    def test_discriminator_output_shape(self):
        """Test discriminator outputs patch-wise predictions."""
        discriminator = PatchGANDiscriminator(in_channels=6, features=64)
        batch_size, height, width = 2, 256, 256
        input_tensor = torch.randn(batch_size, 6, height, width)

        output = discriminator(input_tensor)

        # Output should be (B, 1, H/16, W/16) for 4 stride-2 layers
        assert output.shape[0] == batch_size
        assert output.shape[1] == 1
        assert output.shape[2] == height // 16
        assert output.shape[3] == width // 16

    @pytest.mark.unit
    def test_discriminator_output_range(self):
        """Test discriminator output is in [0, 1] range (Sigmoid activation)."""
        discriminator = PatchGANDiscriminator(use_sigmoid=True)
        input_tensor = torch.randn(1, 6, 256, 256)

        output = discriminator(input_tensor)

        assert output.max() <= 1.0
        assert output.min() >= 0.0


class TestPix2PixModel:
    """Test suite for complete Pix2Pix model."""

    @pytest.mark.unit
    def test_model_forward_pass(self):
        """Test complete model forward pass."""
        model = Pix2PixModel(Config)
        input_tensor = torch.randn(1, 3, 256, 256)

        output = model(input_tensor)

        assert output.shape == input_tensor.shape


@pytest.mark.unit
def test_model_parameters():
    """Test model has trainable parameters."""
    generator = UNetGenerator()
    discriminator = PatchGANDiscriminator()

    gen_params = sum(p.numel() for p in generator.parameters())
    disc_params = sum(p.numel() for p in discriminator.parameters())

    assert gen_params > 0
    assert disc_params > 0
    assert gen_params > disc_params  # Generator usually has more parameters
