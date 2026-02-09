"""
Evaluation metrics for Pix2Pix model.
Includes FID, Inception Score, LPIPS, SSIM, PSNR, etc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import cv2
from skimage.metrics import structural_similarity as skimage_ssim
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr


class MetricComputer:
    """Compute various image quality metrics."""
    
    @staticmethod
    def psnr(generated: torch.Tensor, target: torch.Tensor) -> float:
        """Peak Signal-to-Noise Ratio.
        
        Higher is better. Typical range: 20-50 dB.
        
        Args:
            generated: Generated image tensor [B, C, H, W]
            target: Target image tensor [B, C, H, W]
            
        Returns:
            PSNR score in dB
        """
        # Denormalize from [-1, 1] to [0, 1]
        generated = (generated + 1) / 2
        target = (target + 1) / 2
        
        # Clip to valid range
        generated = torch.clamp(generated, 0, 1)
        target = torch.clamp(target, 0, 1)
        
        # Convert to numpy
        generated = generated.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        
        # Compute PSNR for each image
        psnr_values = []
        for i in range(generated.shape[0]):
            gen_img = np.transpose(generated[i], (1, 2, 0))
            tgt_img = np.transpose(target[i], (1, 2, 0))
            psnr_val = skimage_psnr(tgt_img, gen_img, data_range=1)
            psnr_values.append(psnr_val)
        
        return np.mean(psnr_values)
    
    @staticmethod
    def ssim(generated: torch.Tensor, target: torch.Tensor) -> float:
        """Structural Similarity Index.
        
        Range: -1 to 1. Higher is better.
        
        Args:
            generated: Generated image tensor [B, C, H, W]
            target: Target image tensor [B, C, H, W]
            
        Returns:
            SSIM score
        """
        # Denormalize from [-1, 1] to [0, 1]
        generated = (generated + 1) / 2
        target = (target + 1) / 2
        
        # Clip to valid range
        generated = torch.clamp(generated, 0, 1)
        target = torch.clamp(target, 0, 1)
        
        # Convert to numpy
        generated = generated.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        
        # Compute SSIM for each image
        ssim_values = []
        for i in range(generated.shape[0]):
            gen_img = np.transpose(generated[i], (1, 2, 0))
            tgt_img = np.transpose(target[i], (1, 2, 0))
            
            # Handle multi-channel SSIM
            if gen_img.shape[2] == 3:
                ssim_val = skimage_ssim(tgt_img, gen_img, channel_axis=2, data_range=1)
            else:
                ssim_val = skimage_ssim(tgt_img, gen_img, data_range=1)
            
            ssim_values.append(ssim_val)
        
        return np.mean(ssim_values)
    
    @staticmethod
    def mae(generated: torch.Tensor, target: torch.Tensor) -> float:
        """Mean Absolute Error (L1).
        
        Lower is better.
        
        Args:
            generated: Generated image tensor
            target: Target image tensor
            
        Returns:
            MAE score
        """
        mae = torch.nn.L1Loss()
        return mae(generated, target).item()
    
    @staticmethod
    def mse(generated: torch.Tensor, target: torch.Tensor) -> float:
        """Mean Squared Error (L2).
        
        Lower is better.
        
        Args:
            generated: Generated image tensor
            target: Target image tensor
            
        Returns:
            MSE score
        """
        mse = torch.nn.MSELoss()
        return mse(generated, target).item()
    
    @staticmethod
    def lpips(generated: torch.Tensor, target: torch.Tensor, net: str = 'alex') -> float:
        """Learned Perceptual Image Patch Similarity.
        
        Perceptual distance metric. Lower is better (0-1 range).
        Requires lpips package: pip install lpips
        
        Args:
            generated: Generated image tensor
            target: Target image tensor
            net: Network to use ('alex', 'vgg', 'squeeze')
            
        Returns:
            LPIPS score
        """
        try:
            import lpips as lpips_module
            loss_fn = lpips_module.LPIPS(net=net, verbose=False)
            
            # Normalize to [0, 1]
            generated = (generated + 1) / 2
            target = (target + 1) / 2
            generated = torch.clamp(generated, 0, 1)
            target = torch.clamp(target, 0, 1)
            
            # Compute LPIPS
            with torch.no_grad():
                lpips_val = loss_fn(generated * 2 - 1, target * 2 - 1)
            
            return lpips_val.item()
        
        except ImportError:
            print("LPIPS not installed. Install with: pip install lpips")
            return None


class FrechetInceptionDistance:
    """Fréchet Inception Distance (FID).
    
    Measures the similarity between distributions of generated and real images.
    Better metric than Inception Score as it accounts for both quality and diversity.
    
    Lower scores indicate more realistic images, with the scale roughly:
    - FID < 20: Excellent
    - FID < 30: Good
    - FID < 50: Fair
    """
    
    def __init__(self, device: str = 'cuda'):
        """
        Args:
            device: Device to compute on ('cuda' or 'cpu')
        """
        self.device = device
        self.inception_model = self._load_inception_model()
    
    def _load_inception_model(self):
        """Load pre-trained Inception V3 model."""
        try:
            import torchvision.models as models
            
            inception = models.inception_v3(pretrained=True, transform_input=False)
            inception.eval()
            inception.to(self.device)
            
            # Remove final classification layer
            inception.fc = nn.Identity()
            
            # Freeze parameters
            for param in inception.parameters():
                param.requires_grad = False
            
            return inception
        
        except Exception as e:
            print(f"Failed to load Inception model: {e}")
            return None
    
    def extract_features(self, images: torch.Tensor, batch_size: int = 32) -> np.ndarray:
        """Extract Inception features from images.
        
        Args:
            images: Tensor of images [N, C, H, W]
            batch_size: Batch size for processing
            
        Returns:
            Feature vectors [N, 2048]
        """
        features_list = []
        
        with torch.no_grad():
            for i in range(0, images.shape[0], batch_size):
                batch = images[i:i+batch_size].to(self.device)
                
                # Resize to 299x299 as required by Inception
                batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
                
                # Extract features
                if hasattr(self.inception_model, 'fc'):
                    features = self.inception_model(batch)
                else:
                    # If model has different structure
                    features = self.inception_model.avgpool(
                        self.inception_model.layer4(
                            self.inception_model.layer3(
                                self.inception_model.layer2(
                                    self.inception_model.layer1(batch)
                                )
                            )
                        )
                    )
                    features = features.view(features.size(0), -1)
                
                features_list.append(features.cpu().numpy())
        
        return np.concatenate(features_list, axis=0)
    
    def compute_fid(self, real_images: torch.Tensor, 
                   generated_images: torch.Tensor) -> float:
        """Compute FID between real and generated images.
        
        Args:
            real_images: Real image tensors [N, C, H, W]
            generated_images: Generated image tensors [N, C, H, W]
            
        Returns:
            FID score
        """
        # Extract features
        real_features = self.extract_features(real_images)
        gen_features = self.extract_features(generated_images)
        
        # Compute statistics
        real_mean = np.mean(real_features, axis=0)
        real_cov = np.cov(real_features.T)
        
        gen_mean = np.mean(gen_features, axis=0)
        gen_cov = np.cov(gen_features.T)
        
        # Compute FID
        mean_diff = np.linalg.norm(real_mean - gen_mean) ** 2
        
        # Compute trace of square root of product of covariances
        cov_product = np.linalg.inv(real_cov) @ gen_cov
        trace_term = np.trace(np.linalg.sqrtm(cov_product).real)
        
        fid = mean_diff + np.trace(real_cov) + np.trace(gen_cov) - 2 * trace_term
        
        return fid


class InceptionScore:
    """Inception Score (IS).
    
    Measures both the quality and diversity of generated images.
    Higher scores are better, with typical ranges:
    - IS 3-4: Poor
    - IS 5-6: Decent  
    - IS 7-8: Good
    - IS 9+: Excellent
    """
    
    def __init__(self, device: str = 'cuda'):
        """
        Args:
            device: Device to compute on ('cuda' or 'cpu')
        """
        self.device = device
        self.inception_model = self._load_inception_model()
    
    def _load_inception_model(self):
        """Load pre-trained Inception V3 model."""
        try:
            import torchvision.models as models
            
            inception = models.inception_v3(pretrained=True, transform_input=False)
            inception.eval()
            inception.to(self.device)
            
            # Freeze parameters
            for param in inception.parameters():
                param.requires_grad = False
            
            return inception
        
        except Exception as e:
            print(f"Failed to load Inception model: {e}")
            return None
    
    def compute_is(self, images: torch.Tensor, batch_size: int = 32,
                   splits: int = 10) -> Tuple[float, float]:
        """Compute Inception Score.
        
        Args:
            images: Generated image tensors [N, C, H, W]
            batch_size: Batch size for processing
            splits: Number of splits for computing statistics
            
        Returns:
            (IS mean, IS std)
        """
        scores = []
        
        for split in range(splits):
            part_start = split * len(images) // splits
            part_end = (split + 1) * len(images) // splits
            
            part = images[part_start:part_end]
            
            probs_list = []
            with torch.no_grad():
                for i in range(0, part.shape[0], batch_size):
                    batch = part[i:i+batch_size].to(self.device)
                    
                    # Resize to 299x299
                    batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
                    
                    # Get predictions
                    logits = self.inception_model(batch)
                    probs = F.softmax(logits, dim=1)
                    
                    probs_list.append(probs.cpu().numpy())
            
            probs = np.concatenate(probs_list, axis=0)
            
            # Compute KL divergence
            p_y = np.mean(probs, axis=0)
            kl_div = np.sum(probs * (np.log(probs + 1e-10) - np.log(p_y + 1e-10)), axis=1)
            
            score = np.exp(np.mean(kl_div))
            scores.append(score)
        
        return np.mean(scores), np.std(scores)


class MetricsTracker:
    """Track metrics during training and evaluation."""
    
    def __init__(self):
        self.metrics = {
            'psnr': [],
            'ssim': [],
            'mae': [],
            'mse': [],
            'lpips': []
        }
    
    def update(self, metric_name: str, value: float):
        """Update a metric value."""
        if metric_name in self.metrics:
            self.metrics[metric_name].append(value)
    
    def get_averages(self) -> dict:
        """Get average values for all metrics."""
        averages = {}
        for metric_name, values in self.metrics.items():
            if values:
                averages[metric_name] = np.mean(values)
        return averages
    
    def reset(self):
        """Reset all metrics."""
        for key in self.metrics:
            self.metrics[key] = []


# Export
__all__ = ['MetricComputer', 'FrechetInceptionDistance', 'InceptionScore', 'MetricsTracker']
