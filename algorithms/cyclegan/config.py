"""CycleGAN Configuration."""

class CycleGANConfig:
    """CycleGAN Configuration for unpaired image translation."""
    GENERATOR_PARAMS = 11.4e6  # Dual generators
    DISCRIMINATOR_PARAMS = 3.6e6  # Dual discriminators
    FID_SCORE = 35.2
    INCEPTION_SCORE = 6.1
    LPIPS = 0.267
    SSIM = 0.742
    TRAINING_TIME_HOURS = 42
    INFERENCE_TIME_MS = 310
    DATASET_TYPE = "Unpaired"
    CYCLE_LOSS_WEIGHT = 10.0
    IDENTITY_LOSS_WEIGHT = 0.5
    ADVANTAGES = [
        "Works with unpaired data",
        "No need for aligned image pairs",
        "More flexible for real-world applications"
    ]
    DISADVANTAGES = [
        "Lower photorealism than Pix2Pix",
        "Training less stable",
        "More artifacts in output"
    ]
