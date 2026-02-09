"""
Configuration file for Pix2Pix model training and inference.
Contains all hyperparameters, paths, and settings.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "datasets"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
for dir_path in [DATA_DIR, CHECKPOINTS_DIR, LOGS_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Dataset configuration
class DatasetConfig:
    # Available datasets
    DATASETS = {
        'cityscapes': {
            'name': 'Cityscapes',
            'size': 2975,
            'image_size': (512, 512),
            'domain_a': 'semantic_segmentation',
            'domain_b': 'photo'
        },
        'maps': {
            'name': 'Maps',
            'size': 1100,
            'image_size': (600, 600),
            'domain_a': 'aerial',
            'domain_b': 'map'
        },
        'facades': {
            'name': 'CMP Facades',
            'size': 450,
            'image_size': (512, 512),
            'domain_a': 'segmentation',
            'domain_b': 'facade'
        },
        'edges2shoes': {
            'name': 'Edges2Shoes',
            'size': 50025,
            'image_size': (256, 256),
            'domain_a': 'edge',
            'domain_b': 'shoe'
        },
        'edges2handbags': {
            'name': 'Edges2Handbags',
            'size': 137721,
            'image_size': (256, 256),
            'domain_a': 'edge',
            'domain_b': 'handbag'
        }
    }
    
    DATASET_NAME = 'cityscapes'
    IMAGE_SIZE = (256, 256)
    BATCH_SIZE = 1
    NUM_WORKERS = 4
    TRAIN_TEST_SPLIT = 0.9
    RANDOM_SEED = 42

# Model architecture configuration
class ModelConfig:
    # Generator (U-Net) configuration
    GENERATOR_IN_CHANNELS = 3
    GENERATOR_OUT_CHANNELS = 3
    GENERATOR_FEATURES = 64
    GENERATOR_NORM_TYPE = 'instance'  # 'batch', 'instance', or 'spectral'
    SKIP_CONNECTIONS = True
    
    # Discriminator (PatchGAN) configuration
    DISCRIMINATOR_IN_CHANNELS = 6  # Input + target concatenated
    DISCRIMINATOR_FEATURES = 64
    DISCRIMINATOR_PATCH_SIZE = 70
    DISCRIMINATOR_NORM_TYPE = 'spectral'
    USE_SPECTRAL_NORM = True
    
    # Loss weights
    LAMBDA_GAN = 1.0
    LAMBDA_L1 = 100.0
    LAMBDA_L2 = 0.0
    
    # Activation functions
    GENERATOR_ACTIVATION = 'relu'
    DISCRIMINATOR_ACTIVATION = 'leaky_relu'
    LEAKY_RELU_SLOPE = 0.2

# Training configuration
class TrainingConfig:
    NUM_EPOCHS = 200
    LEARNING_RATE_G = 2e-4
    LEARNING_RATE_D = 2e-4
    BETA1 = 0.5
    BETA2 = 0.999
    OPTIMIZER = 'adam'  # 'adam' or 'sgd'
    
    # Learning rate scheduling
    USE_LR_SCHEDULER = True
    LR_DECAY_START_EPOCH = 100
    LR_DECAY_RATE = 0.5
    
    # Training characteristics
    D_UPDATES_PER_G = 1
    ACCUMULATED_BATCHES = 1
    GRADIENT_CLIP_VALUE = None
    
    # Checkpointing
    CHECKPOINT_INTERVAL = 10
    VALIDATION_INTERVAL = 5
    SAVE_WEIGHT_ONLY = False
    
    # Hardware
    DEVICE = 'cuda'
    MIXED_PRECISION = True
    CUDNN_BENCHMARK = True
    NUM_GPUS = 1

# Data augmentation configuration
class AugmentationConfig:
    RANDOM_FLIP = True
    RANDOM_ROTATION = True
    ROTATION_RANGE = 10  # degrees
    RANDOM_BRIGHTNESS = True
    BRIGHTNESS_RANGE = 0.2
    RANDOM_CONTRAST = True
    CONTRAST_RANGE = 0.2
    RANDOM_HUE = False
    RANDOM_SATURATION = True
    SATURATION_RANGE = 0.2
    GAUSSIAN_NOISE = False
    NOISE_STDDEV = 0.01
    ELASTIC_DEFORMATION = False
    CUTOUT = False
    MIXUP = False

# Evaluation configuration
class EvaluationConfig:
    METRICS = ['fid', 'inception_score', 'lpips', 'ssim', 'psnr']
    FID_BATCH_SIZE = 32
    IS_BATCH_SIZE = 32
    NUM_FID_SAMPLES = 1000
    NUM_INCEPTION_SAMPLES = 1000
    LPIPS_DEVICE = 'cuda'
    LPIPS_NET = 'alex'  # 'alex', 'vgg', or 'squeeze'
    IQA_MODEL = 'brisque'  # For no-reference image quality

# Inference configuration
class InferenceConfig:
    BATCH_SIZE = 8
    OUTPUT_FORMAT = 'png'  # 'png' or 'jpg'
    QUALITY = 95
    SAVE_INTERMEDIATE = False
    TTA_ENABLED = False  # Test-Time Augmentation
    TTA_NUM_AUGMENTATIONS = 4

# Logging and visualization
class LoggingConfig:
    LOG_INTERVAL = 10
    TENSORBOARD = True
    WANDB = False
    SAVE_SAMPLE_OUTPUTS = True
    SAMPLE_OUTPUT_INTERVAL = 50
    NUM_SAMPLES_TO_SAVE = 4
    PLOT_LOSSES = True
    PLOT_METRICS = True

# Consolidate all configs
class Config:
    dataset = DatasetConfig
    model = ModelConfig
    training = TrainingConfig
    augmentation = AugmentationConfig
    evaluation = EvaluationConfig
    inference = InferenceConfig
    logging = LoggingConfig
    
    # Root paths
    PROJECT_ROOT = PROJECT_ROOT
    DATA_DIR = DATA_DIR
    CHECKPOINTS_DIR = CHECKPOINTS_DIR
    LOGS_DIR = LOGS_DIR
    RESULTS_DIR = RESULTS_DIR

# Export
__all__ = ['Config', 'DatasetConfig', 'ModelConfig', 'TrainingConfig', 
           'AugmentationConfig', 'EvaluationConfig', 'InferenceConfig', 'LoggingConfig']
