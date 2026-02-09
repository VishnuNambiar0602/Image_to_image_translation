"""
Quick demo trainer - trains for a few epochs on sample datasets.
Perfect for rapid demonstration and validation.
Use this to show actual training happening with real data!
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import argparse
from tqdm import tqdm

from config import Config
from models import UNetGenerator, PatchGANDiscriminator
from dataset import DatasetFactory
from losses import Pix2PixLoss
from utils import CheckpointManager, set_seed, visualize_batch, save_image
from torchvision.utils import save_image as torchvision_save_image


def demo_train(dataset_name='cityscapes', num_epochs=5, batch_size=1, device='cuda'):
    """
    Quick demo training on sample datasets.
    
    Args:
        dataset_name: Dataset to train on ('cityscapes', 'maps', 'facades', etc)
        num_epochs: Number of epochs to train
        batch_size: Batch size (default 1 for limited memory)
        device: Device to train on
    """
    
    print("\n" + "="*70)
    print(f"PIX2PIX DEMO TRAINING - {dataset_name.upper()}")
    print("="*70)
    
    set_seed(Config.dataset.RANDOM_SEED)
    
    # Create models
    print("\n🏗️ Creating models...")
    generator = UNetGenerator(
        in_channels=3,
        out_channels=3,
        features=Config.model.GENERATOR_FEATURES,
        use_skip=Config.model.SKIP_CONNECTIONS,
        norm_type=Config.model.GENERATOR_NORM_TYPE
    ).to(device)
    
    discriminator = PatchGANDiscriminator(
        in_channels=6,
        features=Config.model.DISCRIMINATOR_FEATURES,
        use_spectral_norm=Config.model.USE_SPECTRAL_NORM
    ).to(device)
    
    # Count parameters
    gen_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    disc_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    
    print(f"✅ Generator: {gen_params:,} parameters")
    print(f"✅ Discriminator: {disc_params:,} parameters")
    print(f"✅ Total: {gen_params + disc_params:,} parameters")
    
    # Create optimizers
    optimizer_g = optim.Adam(
        generator.parameters(),
        lr=Config.training.LEARNING_RATE_G,
        betas=(Config.training.BETA1, Config.training.BETA2)
    )
    
    optimizer_d = optim.Adam(
        discriminator.parameters(),
        lr=Config.training.LEARNING_RATE_D,
        betas=(Config.training.BETA1, Config.training.BETA2)
    )
    
    # Create loss function
    criterion = Pix2PixLoss(
        lambda_gan=Config.model.LAMBDA_GAN,
        lambda_l1=Config.model.LAMBDA_L1
    )
    
    # Create data loader
    print(f"\n📥 Loading {dataset_name} dataset...")
    dataset_path = Config.DATA_DIR / dataset_name
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found at {dataset_path}")
        print("   Run: python create_datasets.py")
        return
    
    train_loader = DatasetFactory.create_dataloader(
        dataset_path=str(dataset_path),
        split='train',
        batch_size=batch_size,
        shuffle=True,
        image_size=Config.dataset.IMAGE_SIZE
    )
    
    print(f"✅ Dataset loaded ({len(train_loader)} batches)")
    
    # Training loop
    print(f"\n🚀 Training for {num_epochs} epochs on {device}...")
    print("-" * 70)
    
    losses_g = []
    losses_d = []
    
    for epoch in range(1, num_epochs + 1):
        generator.train()
        discriminator.train()
        
        epoch_loss_g = 0
        epoch_loss_d = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            source = batch['source'].to(device)
            target = batch['target'].to(device)
            
            # ===== Train Discriminator =====
            optimizer_d.zero_grad()
            
            # Generate fake images
            with torch.no_grad():
                generated = generator(source)
            
            # Create pairs
            real_pair = torch.cat([source, target], dim=1)
            fake_pair = torch.cat([source, generated], dim=1)
            
            # Discriminator outputs
            disc_real = discriminator(real_pair)
            disc_fake = discriminator(fake_pair.detach())
            
            # Loss
            loss_d = criterion.discriminator_loss(disc_real, disc_fake)
            loss_d.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
            
            optimizer_d.step()
            
            # ===== Train Generator =====
            optimizer_g.zero_grad()
            
            # Generate images
            generated = generator(source)
            
            # Create pairs
            fake_pair = torch.cat([source, generated], dim=1)
            
            # Discriminator output
            disc_fake = discriminator(fake_pair)
            
            # Generator loss
            loss_g, loss_dict = criterion.generator_loss(
                generated, target, disc_fake
            )
            
            loss_g.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            
            optimizer_g.step()
            
            # Track losses
            epoch_loss_g += loss_g.item()
            epoch_loss_d += loss_d.item()
            
            losses_g.append(loss_g.item())
            losses_d.append(loss_d.item())
            
            # Update progress bar
            avg_loss_g = epoch_loss_g / (batch_idx + 1)
            avg_loss_d = epoch_loss_d / (batch_idx + 1)
            pbar.set_postfix({
                'Loss_G': f'{avg_loss_g:.4f}',
                'Loss_D': f'{avg_loss_d:.4f}'
            })
            
            # Save sample outputs every 10 batches
            if (batch_idx + 1) % 10 == 0 and batch_idx > 0:
                with torch.no_grad():
                    sample_generated = generator(source[:2])
                    
                    # Save visualization
                    grid = visualize_batch(
                        source[:2], sample_generated[:2], target[:2],
                        num_images=2
                    )
                    
                    viz_dir = Config.LOGS_DIR / dataset_name
                    viz_dir.mkdir(parents=True, exist_ok=True)
                    
                    torchvision_save_image(
                        grid,
                        viz_dir / f'epoch_{epoch:03d}_batch_{batch_idx:04d}.png'
                    )
        
        print(f"\n📊 Epoch {epoch}/{num_epochs} - "
              f"G Loss: {epoch_loss_g/len(train_loader):.4f}, "
              f"D Loss: {epoch_loss_d/len(train_loader):.4f}")
    
    # Save checkpoint
    print("\n💾 Saving checkpoint...")
    checkpoint_dir = Config.CHECKPOINTS_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / f'demo_{dataset_name}_trained.pt'
    
    torch.save({
        'epoch': num_epochs,
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'optimizer_g': optimizer_g.state_dict(),
        'optimizer_d': optimizer_d.state_dict(),
    }, checkpoint_path)
    
    print(f"✅ Checkpoint saved to: {checkpoint_path}")
    
    # Save training results
    print("\n📈 Saving training results...")
    from utils import plot_losses
    plot_losses(losses_g, losses_d, save_path=Config.LOGS_DIR / f'{dataset_name}_losses.png')
    print(f"✅ Loss plot saved!")
    
    print("\n" + "="*70)
    print("✅ DEMO TRAINING COMPLETE!")
    print("="*70)
    
    print(f"""
    📊 Results:
    - Final Generator Loss: {losses_g[-1]:.6f}
    - Final Discriminator Loss: {losses_d[-1]:.6f}
    - Checkpoint: {checkpoint_path}
    - Visualizations: {Config.LOGS_DIR / dataset_name}
    - Loss plot: {Config.LOGS_DIR / f'{dataset_name}_losses.png'}
    
    🚀 Next steps:
    
    1. Run inference on trained model:
       python inference.py \\
           --checkpoint {checkpoint_path} \\
           --input-dir datasets/{dataset_name}/test/source/ \\
           --output-dir results/
    
    2. Train longer for better results:
       python train.py --dataset {dataset_name} --epochs 100
    
    3. Download real datasets for production training:
       python download_datasets.py --info
    """)


def main():
    parser = argparse.ArgumentParser(description='Quick Pix2Pix demo training')
    parser.add_argument('--dataset', type=str, default='cityscapes',
                       choices=['cityscapes', 'maps', 'facades', 'edges2shoes', 'edges2handbags'],
                       help='Dataset to train on')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Get device
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️ Using device: {device}")
    
    # Run training
    demo_train(
        dataset_name=args.dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        device=device
    )


if __name__ == '__main__':
    main()
