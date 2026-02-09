"""
Main training script for Pix2Pix model.
Trains the generator and discriminator in an adversarial framework.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import time
from tqdm import tqdm

from config import Config
from models import Pix2PixModel, UNetGenerator, PatchGANDiscriminator
from dataset import DatasetFactory
from losses import Pix2PixLoss, GANLoss
from metrics import MetricComputer, MetricsTracker
from utils import (
    setup_logging, CheckpointManager, ImageBuffer,
    set_seed, count_parameters, visualize_batch,
    plot_losses, save_results, get_device
)


class Pix2PixTrainer:
    """Trainer class for Pix2Pix model."""
    
    def __init__(self, config, device: str = 'cuda', checkpoint_dir: str = None):
        """
        Args:
            config: Configuration object
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
        """
        self.config = config
        self.device = device
        self.logger = setup_logging(config.LOGS_DIR)
        
        self.logger.info(f"Training on device: {device}")
        
        # Initialize models
        self.generator = UNetGenerator(
            in_channels=config.model.GENERATOR_IN_CHANNELS,
            out_channels=config.model.GENERATOR_OUT_CHANNELS,
            features=config.model.GENERATOR_FEATURES,
            use_skip=config.model.SKIP_CONNECTIONS,
            norm_type=config.model.GENERATOR_NORM_TYPE
        ).to(device)
        
        self.discriminator = PatchGANDiscriminator(
            in_channels=config.model.DISCRIMINATOR_IN_CHANNELS,
            features=config.model.DISCRIMINATOR_FEATURES,
            use_spectral_norm=config.model.USE_SPECTRAL_NORM
        ).to(device)
        
        # Log model info
        self.logger.info(f"Generator parameters: {count_parameters(self.generator):,}")
        self.logger.info(f"Discriminator parameters: {count_parameters(self.discriminator):,}")
        
        # Initialize optimizers
        self.optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=config.training.LEARNING_RATE_G,
            betas=(config.training.BETA1, config.training.BETA2)
        )
        
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=config.training.LEARNING_RATE_D,
            betas=(config.training.BETA1, config.training.BETA2)
        )
        
        # Initialize losses
        self.criterion = Pix2PixLoss(
            lambda_gan=config.model.LAMBDA_GAN,
            lambda_l1=config.model.LAMBDA_L1,
            lambda_l2=config.model.LAMBDA_L2
        )
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir or config.CHECKPOINTS_DIR
        )
        
        # Image buffer for discriminator
        self.image_buffer = ImageBuffer(pool_size=50)
        
        # Metrics tracking
        self.metrics_tracker = MetricsTracker()
        
        # Training history
        self.history = {
            'losses_g': [],
            'losses_d': [],
            'metrics': []
        }
    
    def train_epoch(self, train_loader: DataLoader, epoch: int):
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
        """
        self.generator.train()
        self.discriminator.train()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            source = batch['source'].to(self.device)
            target = batch['target'].to(self.device)
            
            # ===== Train Discriminator =====
            self.optimizer_d.zero_grad()
            
            # Forward pass with generated images
            with torch.no_grad():
                generated = self.generator(source)
            
            # Combine with target
            fake_pair = torch.cat([source, generated], dim=1)
            real_pair = torch.cat([source, target], dim=1)
            
            # Apply image buffer to fake samples
            fake_pair = self.image_buffer.query(fake_pair)
            
            # Discriminator forward pass
            disc_real = self.discriminator(real_pair)
            disc_fake = self.discriminator(fake_pair.detach())
            
            # Discriminator loss
            loss_d = self.criterion.discriminator_loss(disc_real, disc_fake)
            
            loss_d.backward()
            self.optimizer_d.step()
            
            # ===== Train Generator =====
            self.optimizer_g.zero_grad()
            
            # Generate images
            generated = self.generator(source)
            
            # Combine with target
            fake_pair = torch.cat([source, generated], dim=1)
            
            # Discriminator forward pass
            disc_fake = self.discriminator(fake_pair)
            
            # Generator loss
            loss_g, loss_components = self.criterion.generator_loss(
                generated, target, disc_fake
            )
            
            loss_g.backward()
            self.optimizer_g.step()
            
            # Update history
            self.history['losses_g'].append(loss_g.item())
            self.history['losses_d'].append(loss_d.item())
            
            # Update progress bar
            pbar.set_postfix({
                'Loss_G': f'{loss_g.item():.4f}',
                'Loss_D': f'{loss_d.item():.4f}'
            })
            
            # Visualize sample outputs
            if (batch_idx + 1) % self.config.logging.SAMPLE_OUTPUT_INTERVAL == 0:
                sample_viz = visualize_batch(
                    source[:2], generated[:2], target[:2],
                    num_images=2
                )
                
                viz_path = self.config.LOGS_DIR / f'epoch_{epoch:03d}_batch_{batch_idx:04d}.png'
                from torchvision.utils import save_image
                save_image(sample_viz, viz_path)
    
    def validate(self, val_loader: DataLoader, epoch: int) -> dict:
        """Validate the model.
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary with validation metrics
        """
        self.generator.eval()
        self.discriminator.eval()
        
        metrics_dict = {}
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
                source = batch['source'].to(self.device)
                target = batch['target'].to(self.device)
                
                # Generate
                generated = self.generator(source)
                
                # Compute metrics
                psnr = MetricComputer.psnr(generated, target)
                ssim = MetricComputer.ssim(generated, target)
                mae = MetricComputer.mae(generated, target)
                mse = MetricComputer.mse(generated, target)
                
                self.metrics_tracker.update('psnr', psnr)
                self.metrics_tracker.update('ssim', ssim)
                self.metrics_tracker.update('mae', mae)
                self.metrics_tracker.update('mse', mse)
        
        # Get averages
        metrics_dict = self.metrics_tracker.get_averages()
        self.metrics_tracker.reset()
        
        self.logger.info(f"Validation Metrics (Epoch {epoch}): {metrics_dict}")
        
        return metrics_dict
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None,
              num_epochs: int = None):
        """Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            num_epochs: Number of epochs to train
        """
        num_epochs = num_epochs or self.config.training.NUM_EPOCHS
        
        best_metric = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            # Train
            self.train_epoch(train_loader, epoch)
            
            # Validate
            if val_loader is not None and epoch % self.config.training.VALIDATION_INTERVAL == 0:
                val_metrics = self.validate(val_loader, epoch)
                
                # Use MAE as the metric to monitor
                current_metric = val_metrics.get('mae', float('inf'))
                is_best = current_metric < best_metric
                
                if is_best:
                    best_metric = current_metric
                    self.logger.info(f"New best model! MAE: {best_metric:.6f}")
            
            # Save checkpoint
            if epoch % self.config.training.CHECKPOINT_INTERVAL == 0:
                checkpoint_info = {
                    'epoch': epoch,
                    'train_losses_g': self.history['losses_g'][-100:],
                    'train_losses_d': self.history['losses_d'][-100:],
                }
                
                self.checkpoint_manager.save_checkpoint(
                    self.generator, self.discriminator,
                    self.optimizer_g, self.optimizer_d,
                    epoch, checkpoint_info,
                    is_best=is_best if val_loader is not None else False
                )
                
                self.logger.info(f"Checkpoint saved at epoch {epoch}")
            
            # Plot losses
            if epoch % 10 == 0:
                plot_losses(
                    self.history['losses_g'],
                    self.history['losses_d'],
                    save_path=self.config.LOGS_DIR / 'training_losses.png'
                )
        
        self.logger.info("Training completed!")
        
        # Save final results
        save_results(
            {'training_history': self.history},
            self.config.RESULTS_DIR / 'training_results.json'
        )


def main():
    parser = argparse.ArgumentParser(description='Train Pix2Pix model')
    parser.add_argument('--dataset', type=str, default='cityscapes',
                       help='Dataset name')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-4,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Get device
    device = args.device or get_device()
    
    # Create dataloaders
    train_loader = DatasetFactory.create_dataloader(
        dataset_path=Config.DATA_DIR / args.dataset,
        split='train',
        batch_size=args.batch_size
    )
    
    val_loader = DatasetFactory.create_dataloader(
        dataset_path=Config.DATA_DIR / args.dataset,
        split='test',
        batch_size=args.batch_size,
        shuffle=False,
        augment=False
    )
    
    # Create trainer
    trainer = Pix2PixTrainer(Config, device=device)
    
    # Train
    trainer.train(train_loader, val_loader, num_epochs=args.epochs)


if __name__ == '__main__':
    main()
