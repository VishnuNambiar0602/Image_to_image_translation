"""CRN Configuration."""

class CRNConfig:
    """CRN configuration for cascaded refinement networks."""
    GENERATOR_PARAMS = 18.2e6
    FID_SCORE = 41.8
    INCEPTION_SCORE = 5.4
    LPIPS = 0.298
    SSIM = 0.712
    TRAINING_TIME_HOURS = 8  # Faster than GANs
    INFERENCE_TIME_MS = 95  # Faster inference
    DATASET_TYPE = "Paired (Feed-forward)"
    NUM_STAGES = 4
    ADVANTAGES = [
        "Very fast training (8 hours)",
        "Fastest inference (95ms)",
        "No adversarial training complexity",
        "Stable and predictable"
    ]
    DISADVANTAGES = [
        "Lower photorealism than Pix2Pix",
        "Without adversarial loss, misses fine details",
        "FID score 41.8 indicates lower quality"
    ]
