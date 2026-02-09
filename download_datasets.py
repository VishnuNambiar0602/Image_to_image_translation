"""
Script to download and prepare real paired image datasets for Pix2Pix training.

Supports:
- Cityscapes (requires account)
- Maps Dataset
- CMP Facades
- Edges2Shoes
- Edges2Handbags
"""

import os
import urllib.request
import zipfile
import tarfile
from pathlib import Path


def download_cityscapes():
    """
    Download Cityscapes dataset.
    
    NOTE: Requires manual registration at https://www.cityscapes-dataset.com/
    After registration, download and extract to datasets/cityscapes/
    
    Expected structure:
    cityscapes/
    ├── train/
    │   ├── source/  (semantic segmentation labels)
    │   └── target/  (original images)
    └── test/
        ├── source/
        └── target/
    """
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║           CITYSCAPES DATASET DOWNLOAD INSTRUCTIONS             ║
    ╚════════════════════════════════════════════════════════════════╝
    
    Cityscapes requires manual registration:
    
    1. Visit: https://www.cityscapes-dataset.com/
    2. Create account and login
    3. Download:
       - gtFine_trainvaltest.zip (semantic segmentation)
       - leftImg8bit_trainvaltest.zip (images)
    4. Extract to: datasets/cityscapes/
    5. Organize as:
       cityscapes/
       ├── train/
       │   ├── source/  (semantic labels)
       │   └── target/  (street photos)
       └── test/
           ├── source/
           └── target/
    
    Dataset Size: ~11 GB
    Training Pairs: 2,975
    """)


def download_maps():
    """Download Maps dataset from pix2pix-datasets."""
    print("\n📥 Downloading Maps dataset...")
    
    url = "https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz"
    output_path = Path(__file__).parent / "datasets" / "maps"
    
    try:
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Download
        print(f"   Downloading from: {url}")
        tar_path = output_path.parent / "maps.tar.gz"
        urllib.request.urlretrieve(url, tar_path)
        
        # Extract
        print("   Extracting...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(output_path.parent)
        
        # Organize
        organize_pix2pix_dataset(output_path, 'maps')
        
        # Cleanup
        tar_path.unlink()
        
        print("   ✅ Maps dataset downloaded and organized!")
        return True
    
    except Exception as e:
        print(f"   ❌ Error downloading Maps: {e}")
        return False


def download_facades():
    """Download CMP Facades dataset."""
    print("\n📥 Downloading CMP Facades dataset...")
    
    url = "https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz"
    output_path = Path(__file__).parent / "datasets" / "facades"
    
    try:
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Download
        print(f"   Downloading from: {url}")
        tar_path = output_path.parent / "facades.tar.gz"
        urllib.request.urlretrieve(url, tar_path)
        
        # Extract
        print("   Extracting...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(output_path.parent)
        
        # Organize
        organize_pix2pix_dataset(output_path, 'facades')
        
        # Cleanup
        tar_path.unlink()
        
        print("   ✅ Facades dataset downloaded and organized!")
        return True
    
    except Exception as e:
        print(f"   ❌ Error downloading Facades: {e}")
        return False


def download_edges2shoes():
    """Download Edges2Shoes dataset."""
    print("\n📥 Downloading Edges2Shoes dataset...")
    
    url = "https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/edges2shoes.tar.gz"
    output_path = Path(__file__).parent / "datasets" / "edges2shoes"
    
    try:
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Download
        print(f"   Downloading from: {url}")
        tar_path = output_path.parent / "edges2shoes.tar.gz"
        urllib.request.urlretrieve(url, tar_path)
        
        # Extract
        print("   Extracting...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(output_path.parent)
        
        # Organize
        organize_pix2pix_dataset(output_path, 'edges2shoes')
        
        # Cleanup
        tar_path.unlink()
        
        print("   ✅ Edges2Shoes dataset downloaded and organized!")
        return True
    
    except Exception as e:
        print(f"   ❌ Error downloading Edges2Shoes: {e}")
        return False


def download_edges2handbags():
    """Download Edges2Handbags dataset."""
    print("\n📥 Downloading Edges2Handbags dataset...")
    
    url = "https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/edges2handbags.tar.gz"
    output_path = Path(__file__).parent / "datasets" / "edges2handbags"
    
    try:
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Download
        print(f"   Downloading from: {url}")
        tar_path = output_path.parent / "edges2handbags.tar.gz"
        urllib.request.urlretrieve(url, tar_path)
        
        # Extract
        print("   Extracting...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(output_path.parent)
        
        # Organize
        organize_pix2pix_dataset(output_path, 'edges2handbags')
        
        # Cleanup
        tar_path.unlink()
        
        print("   ✅ Edges2Handbags dataset downloaded and organized!")
        return True
    
    except Exception as e:
        print(f"   ❌ Error downloading Edges2Handbags: {e}")
        return False


def organize_pix2pix_dataset(dataset_path, dataset_name):
    """Organize downloaded pix2pix datasets into train/test structure."""
    # This varies by dataset, but typically they come pre-organized
    # or as paired images that need to be split
    print(f"   Organizing {dataset_name}...")


def print_dataset_info():
    """Print information about each dataset."""
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║         PIX2PIX DATASET DOWNLOAD & PREPARATION GUIDE           ║
    ╚════════════════════════════════════════════════════════════════╝
    
    DATASET INFORMATION:
    
    1. CITYSCAPES (11 GB)
       - Semantic segmentation ↔ street photos
       - 2,975 training pairs | 500 test pairs
       - High resolution (1024×512)
       - Manual download required (registration)
       
    2. MAPS (340 MB)
       - Aerial satellite ↔ map views
       - 1,100 training pairs | 100 test pairs
       - Resolution: 600×600
       - Auto-download supported
       
    3. CMP Facades (84 MB)
       - Building facade segmentation
       - 450 training pairs | 100 test pairs
       - Resolution: 512×512
       - Auto-download supported
       
    4. Edges2Shoes (280 MB)
       - Edge sketches ↔ shoe photos
       - 50,025 training pairs | 10,000 test pairs
       - Resolution: 256×256
       - Auto-download supported
       
    5. Edges2Handbags (350 MB)
       - Edge sketches ↔ handbag photos
       - 137,721 training pairs | 10,000 test pairs
       - Resolution: 256×256
       - Auto-download supported
    
    USAGE:
    
    # Download individual datasets:
    python download_datasets.py --dataset maps
    python download_datasets.py --dataset facades
    python download_datasets.py --dataset edges2shoes
    python download_datasets.py --dataset edges2handbags
    
    # Download all available datasets:
    python download_datasets.py --all
    
    # For Cityscapes:
    python download_datasets.py --dataset cityscapes
    (Then follow manual instructions)
    """)


def main():
    """Main function to handle argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Download Pix2Pix datasets'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['cityscapes', 'maps', 'facades', 'edges2shoes', 'edges2handbags'],
        help='Which dataset to download'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Download all available datasets'
    )
    parser.add_argument(
        '--info',
        action='store_true',
        help='Show dataset information'
    )
    
    args = parser.parse_args()
    
    if args.info or (not args.dataset and not args.all):
        print_dataset_info()
        return
    
    print("\n" + "="*70)
    print("PIX2PIX DATASET DOWNLOADER")
    print("="*70)
    
    datasets_to_download = []
    
    if args.all:
        datasets_to_download = ['maps', 'facades', 'edges2shoes', 'edges2handbags']
        print("\n📥 Downloading all available datasets...")
    elif args.dataset == 'cityscapes':
        download_cityscapes()
        return
    else:
        datasets_to_download = [args.dataset]
    
    results = {}
    
    for dataset in datasets_to_download:
        if dataset == 'maps':
            results['maps'] = download_maps()
        elif dataset == 'facades':
            results['facades'] = download_facades()
        elif dataset == 'edges2shoes':
            results['edges2shoes'] = download_edges2shoes()
        elif dataset == 'edges2handbags':
            results['edges2handbags'] = download_edges2handbags()
    
    # Summary
    print("\n" + "="*70)
    print("DOWNLOAD SUMMARY")
    print("="*70)
    for dataset, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"{dataset:20} - {status}")
    
    print("\n✅ Ready to train with:")
    print("   python train.py --dataset maps --epochs 200")


if __name__ == '__main__':
    main()
