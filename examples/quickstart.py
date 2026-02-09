"""Quick start examples showing how to use Pix2Pix."""

# Example 1: Basic model creation
from src.pix2pix.models import UNetGenerator, PatchGANDiscriminator
from src.pix2pix.config import Config
import torch

def example_1_create_model():
    """Create generator and discriminator."""
    generator = UNetGenerator(
        in_channels=3,
        out_channels=3,
        features=64,
        use_skip=True
    )
    discriminator = PatchGANDiscriminator(
        in_channels=6,
        features=64,
        use_spectral_norm=True
    )
    print(f"Generator params: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}")


# Example 2: Forward pass
def example_2_forward_pass():
    """Perform forward pass through models."""
    generator = UNetGenerator()
    discriminator = PatchGANDiscriminator()
    
    source = torch.randn(2, 3, 256, 256)
    target = generator(source)
    
    # Concatenate source and target for discriminator
    concatenated = torch.cat([source, target], dim=1)
    discrimination = discriminator(concatenated)
    
    print(f"Source shape: {source.shape}")
    print(f"Generated target shape: {target.shape}")
    print(f"Discriminator output shape: {discrimination.shape}")


# Example 3: Load dataset
def example_3_load_dataset():
    """Load and iterate over dataset."""
    from src.pix2pix.dataset import PairedImageDataset
    from pathlib import Path
    
    dataset_path = Path("datasets/cityscapes")
    if dataset_path.exists():
        dataset = PairedImageDataset(
            dataset_path / "train",
            image_size=256,
            augment=True
        )
        print(f"Dataset size: {len(dataset)}")
        
        sample = dataset[0]
        print(f"Source shape: {sample['source'].shape}")
        print(f"Target shape: {sample['target'].shape}")


# Example 4: Training setup
def example_4_training_setup():
    """Setup training components."""
    from src.pix2pix.losses import Pix2PixLoss
    from src.pix2pix.models import Pix2PixModel
    
    model = Pix2PixModel(Config)
    criterion = Pix2PixLoss(config=Config)
    
    gen_optimizer = torch.optim.Adam(
        model.generator.parameters(),
        lr=Config.training.LEARNING_RATE_G,
        betas=(Config.training.BETA1, Config.training.BETA2)
    )
    
    disc_optimizer = torch.optim.Adam(
        model.discriminator.parameters(),
        lr=Config.training.LEARNING_RATE_D,
        betas=(Config.training.BETA1, Config.training.BETA2)
    )
    
    print("✅ Training setup complete")
    print(f"  Generator LR: {Config.training.LEARNING_RATE_G}")
    print(f"  Discriminator LR: {Config.training.LEARNING_RATE_D}")


# Example 5: Inference
def example_5_inference():
    """Load checkpoint and run inference."""
    from src.pix2pix.inference import Pix2PixInference
    from pathlib import Path
    
    checkpoint_path = Path("checkpoints/demo_cityscapes_trained.pt")
    if checkpoint_path.exists():
        inference = Pix2PixInference(checkpoint_path=str(checkpoint_path))
        
        # Translate single image
        image_path = Path("datasets/cityscapes/test/source")
        if image_path.exists():
            images = list(image_path.glob("*.jpg"))
            if images:
                result = inference.translate_single(str(images[0]))
                print(f"Translated image shape: {result.shape}")


if __name__ == "__main__":
    print("=" * 60)
    print("Pix2Pix Quick Start Examples")
    print("=" * 60)
    
    print("\n[Example 1] Create Models")
    print("-" * 60)
    example_1_create_model()
    
    print("\n[Example 2] Forward Pass")
    print("-" * 60)
    example_2_forward_pass()
    
    print("\n[Example 3] Load Dataset")
    print("-" * 60)
    example_3_load_dataset()
    
    print("\n[Example 4] Setup Training")
    print("-" * 60)
    example_4_training_setup()
    
    print("\n[Example 5] Run Inference")
    print("-" * 60)
    example_5_inference()
    
    print("\n" + "=" * 60)
    print("✅ All examples completed!")
    print("=" * 60)
