"""PSPNet Configuration."""

class PSPNetConfig:
    """PSPNet configuration for semantic segmentation with photorealism."""
    MODEL_PARAMS = 44.5e6
    FID_SCORE = 47.2
    INCEPTION_SCORE = 4.8
    LPIPS = 0.341
    SSIM = 0.654
    TRAINING_TIME_HOURS = 24
    INFERENCE_TIME_MS = 150
    DATASET_TYPE = "Semantic Segmentation"
    NUM_CLASSES = 150
    ADVANTAGES = [
        "Fastest training (24 hours)",
        "Interpretable outputs (segmentation maps)",
        "Good for scene understanding",
        "Lowest inference time"
    ]
    DISADVANTAGES = [
        "Lacks adversarial feedback",
        "Results are blurry",
        "Lower perceptual quality",
        "Higher FID score (worse)"
    ]
