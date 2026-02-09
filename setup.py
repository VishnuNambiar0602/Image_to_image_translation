from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pix2pix-gan",
    version="1.0.0",
    author="GAN_CIA Research",
    description="Pix2Pix: Image-to-Image Translation with Conditional GANs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pix2pix",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.21.0",
        "Pillow>=8.0.0",
        "scikit-image>=0.18.0",
        "opencv-python>=4.5.0",
        "matplotlib>=3.3.0",
        "tqdm>=4.50.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
        "evaluation": [
            "lpips>=0.1.4",
            "tensorboard>=2.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pix2pix-train=train:main",
            "pix2pix-infer=inference:main",
        ],
    },
)
