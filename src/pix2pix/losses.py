"""
Loss functions for Pix2Pix training.
Includes adversarial loss, L1 loss, and other metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class GANLoss(nn.Module):
    """Adversarial loss for GAN training.
    
    Uses Binary Cross Entropy loss for discriminator outputs.
    """
    
    def __init__(self, use_lsgan: bool = False, target_real_label: float = 1.0,
                 target_fake_label: float = 0.0):
        """
        Args:
            use_lsgan: If True, use Least Squares GAN loss
            target_real_label: Label for real images (usually 1.0)
            target_fake_label: Label for fake images (usually 0.0)
        """
        super().__init__()
        
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        
        self.use_lsgan = use_lsgan
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()
    
    def get_target_tensor(self, input: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        """Get target tensor for loss computation."""
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        
        return target_tensor.expand_as(input)
    
    def __call__(self, input: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        """
        Args:
            input: Discriminator output
            target_is_real: Whether target is real (True) or fake (False)
            
        Returns:
            Loss value
        """
        target_tensor = self.get_target_tensor(input, target_is_real)
        loss = self.loss(input, target_tensor)
        return loss


class L1Loss(nn.Module):
    """L1 Reconstruction Loss.
    
    Encourages the generator to produce outputs that are close to
    the ground truth target in terms of pixel values.
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.loss = nn.L1Loss(reduction=reduction)
    
    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(generated, target)


class L2Loss(nn.Module):
    """L2 Reconstruction Loss (MSE).
    
    Penalizes larger differences more than L1 loss.
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.loss = nn.MSELoss(reduction=reduction)
    
    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(generated, target)


class PerceptualLoss(nn.Module):
    """Perceptual Loss using pre-trained VGG16.
    
    Compares intermediate feature maps rather than pixel values,
    encouraging perceptually similar outputs.
    """
    
    def __init__(self, layer: str = 'relu3_4'):
        """
        Args:
            layer: Which VGG layer to use ('relu1_2', 'relu2_2', 'relu3_4', etc.)
        """
        super().__init__()
        
        # Load pre-trained VGG16
        vgg = torchvision.models.vgg16(pretrained=True)
        
        # Extract features up to the specified layer
        layer_map = {
            'relu1_1': 1, 'relu1_2': 3,
            'relu2_1': 6, 'relu2_2': 8,
            'relu3_1': 11, 'relu3_2': 13, 'relu3_3': 15, 'relu3_4': 17,
            'relu4_1': 20, 'relu4_2': 22, 'relu4_3': 24, 'relu4_4': 26,
            'relu5_1': 29, 'relu5_2': 31, 'relu5_3': 33, 'relu5_4': 35
        }
        
        max_layer = layer_map.get(layer, 17)
        self.features = vgg.features[:max_layer+1]
        
        # Freeze parameters
        for param in self.features.parameters():
            param.requires_grad = False
        
        self.loss = nn.L1Loss()
    
    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            generated: Generated image tensor
            target: Target image tensor
            
        Returns:
            Perceptual loss value
        """
        features_gen = self.features(generated)
        features_target = self.features(target)
        return self.loss(features_gen, features_target)


class StyleLoss(nn.Module):
    """Style Loss using Gram matrices.
    
    Encourages similar perceptual style between generated and target images.
    """
    
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
    
    def compute_gram_matrix(self, features: torch.Tensor) -> torch.Tensor:
        """Compute Gram matrix for given features."""
        batch_size, channels, height, width = features.size()
        
        # Reshape features
        features = features.view(batch_size, channels, -1)
        
        # Compute Gram matrix
        gram = torch.bmm(features, features.transpose(1, 2))
        gram = gram / (channels * height * width)
        
        return gram
    
    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            generated: Generated image features
            target: Target image features
            
        Returns:
            Style loss value
        """
        gram_gen = self.compute_gram_matrix(generated)
        gram_target = self.compute_gram_matrix(target)
        
        return self.mse_loss(gram_gen, gram_target)


class Pix2PixLoss(nn.Module):
    """Complete loss function for Pix2Pix training.
    
    Combines adversarial loss and L1/L2 reconstruction loss.
    """
    
    def __init__(self, lambda_gan: float = 1.0, lambda_l1: float = 100.0,
                 lambda_l2: float = 0.0, use_lsgan: bool = False):
        """
        Args:
            lambda_gan: Weight for adversarial loss
            lambda_l1: Weight for L1 reconstruction loss
            lambda_l2: Weight for L2 reconstruction loss
            use_lsgan: Whether to use Least Squares GAN loss
        """
        super().__init__()
        
        self.lambda_gan = lambda_gan
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        
        self.gan_loss = GANLoss(use_lsgan=use_lsgan)
        self.l1_loss = L1Loss() if lambda_l1 > 0 else None
        self.l2_loss = L2Loss() if lambda_l2 > 0 else None
    
    def discriminator_loss(self, disc_real: torch.Tensor, 
                          disc_fake: torch.Tensor) -> torch.Tensor:
        """
        Discriminator loss.
        
        Args:
            disc_real: Discriminator output on real image pairs
            disc_fake: Discriminator output on generated image pairs
            
        Returns:
            Total discriminator loss
        """
        loss_real = self.gan_loss(disc_real, target_is_real=True)
        loss_fake = self.gan_loss(disc_fake, target_is_real=False)
        
        return (loss_real + loss_fake) * 0.5
    
    def generator_loss(self, generated: torch.Tensor, target: torch.Tensor,
                      disc_fake: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Generator loss.
        
        Args:
            generated: Generated image from generator
            target: Target image
            disc_fake: Discriminator output on generated image pair
            
        Returns:
            Total generator loss and loss components dictionary
        """
        losses = {}
        
        # Adversarial loss
        loss_gan = self.gan_loss(disc_fake, target_is_real=True)
        losses['gan'] = loss_gan.item()
        
        total_loss = self.lambda_gan * loss_gan
        
        # L1 Loss
        if self.l1_loss is not None:
            loss_l1 = self.l1_loss(generated, target)
            losses['l1'] = loss_l1.item()
            total_loss = total_loss + self.lambda_l1 * loss_l1
        
        # L2 Loss
        if self.l2_loss is not None:
            loss_l2 = self.l2_loss(generated, target)
            losses['l2'] = loss_l2.item()
            total_loss = total_loss + self.lambda_l2 * loss_l2
        
        losses['total'] = total_loss.item()
        
        return total_loss, losses


# Export
__all__ = ['GANLoss', 'L1Loss', 'L2Loss', 'PerceptualLoss', 
           'StyleLoss', 'Pix2PixLoss']
