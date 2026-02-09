"""
Script to create sample datasets for demonstration and testing.
Creates minimal but realistic paired images for each Pix2Pix domain.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from pathlib import Path


def create_sample_datasets():
    """Create sample dataset directories with dummy image pairs."""
    
    datasets_root = Path(__file__).parent / 'datasets'
    
    # Dataset configurations
    datasets_config = {
        'cityscapes': {
            'splits': {'train': 10, 'test': 3},  # 10 training, 3 test pairs
            'size': (256, 256),
            'description': 'Semantic Segmentation ↔ Street Photos'
        },
        'maps': {
            'splits': {'train': 8, 'test': 2},
            'size': (256, 256),
            'description': 'Aerial ↔ Map Translation'
        },
        'facades': {
            'splits': {'train': 6, 'test': 2},
            'size': (256, 256),
            'description': 'Building Segmentation ↔ Facade'
        },
        'edges2shoes': {
            'splits': {'train': 15, 'test': 3},
            'size': (256, 256),
            'description': 'Edge Sketch ↔ Shoe Photo'
        },
        'edges2handbags': {
            'splits': {'train': 15, 'test': 3},
            'size': (256, 256),
            'description': 'Edge Sketch ↔ Handbag Photo'
        }
    }
    
    print("=" * 70)
    print("CREATING SAMPLE DATASETS FOR PIX2PIX")
    print("=" * 70)
    
    for dataset_name, config in datasets_config.items():
        print(f"\n📁 Creating {dataset_name.upper()} dataset...")
        print(f"   {config['description']}")
        
        dataset_path = datasets_root / dataset_name
        
        for split, count in config['splits'].items():
            source_dir = dataset_path / split / 'source'
            target_dir = dataset_path / split / 'target'
            
            source_dir.mkdir(parents=True, exist_ok=True)
            target_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"   Creating {count} {split} pairs...")
            
            for i in range(count):
                if dataset_name == 'cityscapes':
                    source_img = create_semantic_segmentation(config['size'])
                    target_img = create_street_photo(config['size'])
                
                elif dataset_name == 'maps':
                    source_img = create_aerial_image(config['size'])
                    target_img = create_map_image(config['size'])
                
                elif dataset_name == 'facades':
                    source_img = create_building_segmentation(config['size'])
                    target_img = create_facade_photo(config['size'])
                
                elif dataset_name == 'edges2shoes':
                    source_img = create_edge_sketch(config['size'])
                    target_img = create_shoe_photo(config['size'])
                
                elif dataset_name == 'edges2handbags':
                    source_img = create_edge_sketch(config['size'])
                    target_img = create_handbag_photo(config['size'])
                
                # Save images
                source_path = source_dir / f'{dataset_name}_{split}_{i:04d}.jpg'
                target_path = target_dir / f'{dataset_name}_{split}_{i:04d}.jpg'
                
                source_img.save(source_path, quality=95)
                target_img.save(target_path, quality=95)
        
        print(f"   ✅ {dataset_name} dataset created!")
    
    print("\n" + "=" * 70)
    print("✅ ALL SAMPLE DATASETS CREATED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nDatasets saved to: {datasets_root}")
    print("\nDataset Statistics:")
    for dataset_name, config in datasets_config.items():
        total = sum(config['splits'].values())
        print(f"  • {dataset_name:20} - {total:2d} total pairs")
    
    print("\nYou can now train with:")
    print("  python train.py --dataset cityscapes --epochs 10")


def create_semantic_segmentation(size):
    """Create synthetic semantic segmentation mask."""
    height, width = size
    img = Image.new('RGB', (width, height), color='white')
    
    # Add some colored regions to represent segmentation
    pixels = img.load()
    
    colors = [
        (128, 64, 128),   # Purple - road
        (244, 35, 232),   # Pink - sidewalk
        (70, 70, 70),     # Gray - building
        (102, 102, 156),  # Blue - wall
        (190, 153, 153),  # Brown - fence
        (0, 0, 142),      # Dark blue - car
    ]
    
    # Fill regions with random colors
    for y in range(height):
        for x in range(width):
            if x < width // 2:
                color = colors[0]
            elif y < height // 3:
                color = colors[2]
            else:
                color = colors[1]
            pixels[x, y] = color
    
    return img


def create_street_photo(size):
    """Create synthetic street scene photo."""
    img = Image.new('RGB', size, color=(100, 150, 200))  # Sky blue
    pixels = img.load()
    height, width = size
    
    # Add ground (bottom half)
    for y in range(height // 2, height):
        for x in range(width):
            gray = 100 + np.random.randint(-10, 10)
            pixels[x, y] = (gray, gray, gray)
    
    # Add buildings (top half with color variation)
    for y in range(0, height // 3):
        for x in range(width):
            r = 150 + np.random.randint(-30, 30)
            g = 140 + np.random.randint(-30, 30)
            b = 130 + np.random.randint(-30, 30)
            pixels[x, y] = (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))
    
    return img


def create_aerial_image(size):
    """Create synthetic aerial/satellite image."""
    img = Image.new('RGB', size, color=(34, 139, 34))  # Green (grass)
    pixels = img.load()
    height, width = size
    
    # Add roads (gray lines)
    for y in range(height):
        for x in range(width):
            if (x % 80 < 10 or y % 80 < 10):
                pixels[x, y] = (150, 150, 150)  # Road gray
            elif (x % 120 < 20 and y % 120 < 20):
                pixels[x, y] = (100, 149, 237)  # Water blue
    
    return img


def create_map_image(size):
    """Create synthetic map representation."""
    img = Image.new('RGB', size, color=(200, 200, 150))  # Map beige
    pixels = img.load()
    height, width = size
    
    # Add colored regions
    for y in range(height):
        for x in range(width):
            if (x % 100 < 15 or y % 100 < 15):
                pixels[x, y] = (0, 0, 0)  # Black roads
            elif (x % 120 < 25 and y % 120 < 25):
                pixels[x, y] = (0, 100, 200)  # Blue water
    
    return img


def create_building_segmentation(size):
    """Create building facade segmentation mask."""
    img = Image.new('RGB', size, color=(200, 200, 200))
    pixels = img.load()
    height, width = size
    
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    
    # Create grid pattern of different materials
    for y in range(height):
        for x in range(width):
            grid_x = (x // 32) % 2
            grid_y = (y // 32) % 2
            color = colors[(grid_x + grid_y) % len(colors)]
            pixels[x, y] = color
    
    return img


def create_facade_photo(size):
    """Create synthetic building facade photo."""
    img = Image.new('RGB', size, color=(180, 170, 160))  # Building color
    pixels = img.load()
    height, width = size
    
    # Add window pattern
    for y in range(height):
        for x in range(width):
            if (x % 40 < 20 and y % 40 < 20):
                pixels[x, y] = (100, 150, 200)  # Window blue
            else:
                pixels[x, y] = (150 + np.random.randint(-20, 20), 
                              140 + np.random.randint(-20, 20), 
                              130 + np.random.randint(-20, 20))
    
    return img


def create_edge_sketch(size):
    """Create synthetic edge sketch drawing."""
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw edge lines
    line_color = 'black'
    line_width = 2
    
    # Draw some contours
    draw.polygon(
        [(50, 50), (200, 50), (220, 150), (100, 200), (20, 150)],
        outline=line_color,
        width=line_width
    )
    
    # Add more edge details
    draw.ellipse([150, 80, 200, 130], outline=line_color, width=line_width)
    
    return img


def create_shoe_photo(size):
    """Create synthetic shoe product photo."""
    img = Image.new('RGB', size, color=(240, 240, 240))
    pixels = img.load()
    height, width = size
    
    # Create shoe-like shape with color
    for y in range(height):
        for x in range(width):
            # Distance from center
            dx = x - width // 2
            dy = y - height // 2
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist < 80:
                # Shoe color
                pixels[x, y] = (200, 100, 50)
            elif dist < 90:
                # Sole
                pixels[x, y] = (0, 0, 0)
    
    return img


def create_handbag_photo(size):
    """Create synthetic handbag product photo."""
    img = Image.new('RGB', size, color=(250, 250, 250))
    pixels = img.load()
    height, width = size
    
    # Create handbag shape
    for y in range(height):
        for x in range(width):
            dx = x - width // 2
            dy = y - height // 2
            
            # Main bag body
            if dy < 100 and abs(dx) < 60:
                pixels[x, y] = (139, 69, 19)  # Brown leather
            # Handle
            elif 80 < dy < 120 and abs(dx) < 40:
                pixels[x, y] = (101, 50, 15)  # Darker brown
    
    return img


if __name__ == '__main__':
    create_sample_datasets()
