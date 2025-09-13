"""
Setup script to create organized folder structure for the Wall AR Pipeline.
"""

import os
import shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

def create_folder_structure():
    """Create organized input and output folder structure."""
    
    # Define folder structure
    folders = [
        "input",
        "input/walls", 
        "input/products",
        "input/products/tvs",
        "input/products/paintings", 
        "input/products/frames",
        "output",
        "output/results",
        "output/debug"
    ]
    
    # Create folders
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created folder: {folder}")

def create_sample_wall_images():
    """Create sample wall/room images."""
    
    def create_living_room(width=800, height=600):
        """Create a living room scene."""
        room = np.ones((height, width, 3), dtype=np.uint8) * 200
        
        # Wall region (light beige)
        wall_region = room[50:height-120, 100:width-100]
        wall_region[:, :] = [235, 225, 210]
        
        # Add furniture
        # Sofa (dark brown)
        room[height-120:height-40, 50:300] = [101, 67, 33]
        # Coffee table (wood)
        room[height-80:height-60, 150:250] = [139, 119, 101]
        # Side table
        room[height-100:height-40, width-200:width-120] = [139, 119, 101]
        
        # Floor (wood)
        room[height-40:height, :] = [160, 140, 120]
        
        # Add lighting gradient
        for i in range(height):
            brightness = 0.7 + 0.5 * (i / height)
            room[i, :] = np.clip(room[i, :] * brightness, 0, 255)
        
        return room.astype(np.uint8)
    
    def create_bedroom(width=700, height=500):
        """Create a bedroom scene."""
        room = np.ones((height, width, 3), dtype=np.uint8) * 180
        
        # Wall (soft blue-gray)
        wall_region = room[40:height-100, 80:width-80]
        wall_region[:, :] = [220, 220, 235]
        
        # Bed
        room[height-100:height-20, 100:400] = [80, 60, 40]
        # Nightstand
        room[height-80:height-20, 420:500] = [120, 100, 80]
        
        # Floor (carpet)
        room[height-20:height, :] = [140, 130, 125]
        
        return room.astype(np.uint8)
    
    def create_office(width=900, height=650):
        """Create an office scene."""
        room = np.ones((height, width, 3), dtype=np.uint8) * 220
        
        # Wall (white)
        wall_region = room[60:height-100, 120:width-120]
        wall_region[:, :] = [245, 245, 240]
        
        # Desk
        room[height-100:height-60, 200:600] = [100, 80, 60]
        # Chair
        room[height-80:height-40, 350:450] = [60, 60, 60]
        # Bookshelf
        room[100:height-100, width-120:width-40] = [120, 100, 80]
        
        # Floor (tile)
        room[height-40:height, :] = [200, 200, 195]
        
        return room.astype(np.uint8)
    
    # Create and save wall images
    walls = {
        "living_room_1.jpg": create_living_room(800, 600),
        "living_room_2.jpg": create_living_room(1024, 768),
        "bedroom_1.jpg": create_bedroom(700, 500),
        "bedroom_2.jpg": create_bedroom(900, 600),
        "office_1.jpg": create_office(900, 650),
        "office_2.jpg": create_office(1200, 800)
    }
    
    walls_dir = Path("input/walls")
    for filename, image in walls.items():
        output_path = walls_dir / filename
        cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(f"✓ Created wall image: {output_path}")

def create_sample_product_images():
    """Create sample product images."""
    
    def create_modern_tv(width=300, height=169):
        """Create a modern TV image."""
        tv = np.zeros((height, width, 4), dtype=np.uint8)
        
        # Bezel (dark gray)
        bezel = 8
        tv[bezel:-bezel, bezel:-bezel] = [25, 25, 25, 255]
        
        # Screen (black with blue tint)
        screen = 15
        tv[screen:-screen, screen:-screen] = [5, 10, 20, 255]
        
        # Add screen content (gradient)
        screen_region = tv[screen:-screen, screen:-screen]
        for i in range(screen_region.shape[0]):
            intensity = int(30 + 60 * (i / screen_region.shape[0]))
            screen_region[i, :] = [intensity//4, intensity//2, intensity, 255]
        
        # Add brand logo area (slightly lighter)
        logo_h = height // 8
        tv[-bezel-logo_h:-bezel, width//2-30:width//2+30] = [40, 40, 40, 255]
        
        return tv
    
    def create_vintage_tv(width=280, height=210):
        """Create a vintage/CRT TV."""
        tv = np.zeros((height, width, 4), dtype=np.uint8)
        
        # Body (beige/cream)
        tv[:, :] = [200, 190, 170, 255]
        
        # Screen (curved, black)
        screen_margin = 40
        screen_region = tv[screen_margin:-screen_margin-60, screen_margin:-screen_margin]
        screen_region[:, :] = [10, 10, 15, 255]
        
        # Control panel
        controls_h = 50
        tv[-controls_h:, :] = [180, 170, 150, 255]
        
        return tv
    
    def create_abstract_painting(width=200, height=250):
        """Create an abstract painting."""
        painting = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Frame (gold)
        frame = 20
        painting[:frame, :] = [180, 140, 60]
        painting[-frame:, :] = [180, 140, 60]
        painting[:, :frame] = [180, 140, 60]
        painting[:, -frame:] = [180, 140, 60]
        
        # Canvas (off-white)
        canvas = painting[frame:-frame, frame:-frame]
        canvas[:, :] = [250, 245, 240]
        
        # Add abstract art
        center_y, center_x = canvas.shape[0] // 2, canvas.shape[1] // 2
        
        # Blue shapes
        cv2.circle(painting, (center_x + frame, center_y + frame - 30), 40, (100, 150, 220), -1)
        cv2.rectangle(painting, 
                     (center_x - 30 + frame, center_y + 20 + frame),
                     (center_x + 30 + frame, center_y + 60 + frame),
                     (220, 100, 100), -1)
        
        # Yellow accent
        cv2.ellipse(painting, (center_x + frame + 20, center_y + frame + 10), 
                   (25, 15), 45, 0, 360, (220, 200, 80), -1)
        
        return painting
    
    def create_landscape_painting(width=300, height=200):
        """Create a landscape painting."""
        painting = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Frame (dark wood)
        frame = 15
        painting[:frame, :] = [80, 60, 40]
        painting[-frame:, :] = [80, 60, 40]
        painting[:, :frame] = [80, 60, 40]
        painting[:, -frame:] = [80, 60, 40]
        
        # Canvas
        canvas = painting[frame:-frame, frame:-frame]
        
        # Sky (blue gradient)
        for i in range(canvas.shape[0] // 2):
            intensity = 150 + int(50 * (i / (canvas.shape[0] // 2)))
            canvas[i, :] = [intensity//3, intensity//2, intensity]
        
        # Ground (green)
        canvas[canvas.shape[0]//2:, :] = [60, 120, 40]
        
        # Add simple landscape elements
        # Mountains
        pts = np.array([[50, canvas.shape[0]//2], [100, canvas.shape[0]//2-30], 
                       [150, canvas.shape[0]//2]], np.int32)
        cv2.fillPoly(painting[frame:-frame, frame:-frame], [pts], (100, 100, 120))
        
        return painting
    
    def create_photo_frame(width=180, height=240):
        """Create a photo frame."""
        frame_img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Frame (silver/gray)
        frame_thickness = 25
        frame_img[:frame_thickness, :] = [160, 160, 170]
        frame_img[-frame_thickness:, :] = [160, 160, 170]
        frame_img[:, :frame_thickness] = [160, 160, 170]
        frame_img[:, -frame_thickness:] = [160, 160, 170]
        
        # Photo area (family photo simulation)
        photo = frame_img[frame_thickness:-frame_thickness, frame_thickness:-frame_thickness]
        photo[:, :] = [220, 210, 200]  # Sepia tone
        
        # Add simple photo content
        # Faces (circles)
        cv2.circle(frame_img, (width//2 - 20, height//2 - 20), 15, (200, 180, 160), -1)
        cv2.circle(frame_img, (width//2 + 20, height//2 - 20), 15, (200, 180, 160), -1)
        cv2.circle(frame_img, (width//2, height//2 + 20), 12, (200, 180, 160), -1)
        
        return frame_img
    
    # Create TV images
    tvs_dir = Path("input/products/tvs")
    tv_images = {
        "modern_tv_55inch.png": create_modern_tv(300, 169),
        "modern_tv_65inch.png": create_modern_tv(400, 225),
        "vintage_tv.png": create_vintage_tv(280, 210)
    }
    
    for filename, image in tv_images.items():
        output_path = tvs_dir / filename
        if image.shape[2] == 4:  # RGBA
            cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA))
        else:  # RGB
            cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(f"✓ Created TV image: {output_path}")
    
    # Create painting images
    paintings_dir = Path("input/products/paintings")
    painting_images = {
        "abstract_art.png": create_abstract_painting(200, 250),
        "landscape.png": create_landscape_painting(300, 200),
        "modern_abstract.png": create_abstract_painting(250, 300)
    }
    
    for filename, image in painting_images.items():
        output_path = paintings_dir / filename
        cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(f"✓ Created painting image: {output_path}")
    
    # Create frame images
    frames_dir = Path("input/products/frames")
    frame_images = {
        "family_photo.png": create_photo_frame(180, 240),
        "portrait_frame.png": create_photo_frame(200, 280)
    }
    
    for filename, image in frame_images.items():
        output_path = frames_dir / filename
        cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(f"✓ Created frame image: {output_path}")

def move_existing_samples():
    """Move existing sample images to new structure."""
    sample_images_dir = Path("sample_images")
    
    if sample_images_dir.exists():
        print("\n📁 Moving existing sample images...")
        
        # Move room images to walls folder
        for room_file in sample_images_dir.glob("room*.jpg"):
            dest = Path("input/walls") / room_file.name
            if room_file.exists():
                shutil.copy2(room_file, dest)
                print(f"✓ Moved {room_file.name} to input/walls/")
        
        # Move TV images to TVs folder
        for tv_file in sample_images_dir.glob("tv*.png"):
            dest = Path("input/products/tvs") / tv_file.name
            if tv_file.exists():
                shutil.copy2(tv_file, dest)
                print(f"✓ Moved {tv_file.name} to input/products/tvs/")
        
        # Move painting images to paintings folder
        for painting_file in sample_images_dir.glob("painting*.png"):
            dest = Path("input/products/paintings") / painting_file.name
            if painting_file.exists():
                shutil.copy2(painting_file, dest)
                print(f"✓ Moved {painting_file.name} to input/products/paintings/")

def create_readme_files():
    """Create README files for each folder."""
    
    readme_contents = {
        "input/README.md": """# Input Images

This folder contains all input images organized by category:

- `walls/` - Room images with visible walls for product placement
- `products/` - Product images to be placed on walls
  - `tvs/` - Television images (preferably with transparent backgrounds)
  - `paintings/` - Painting and artwork images
  - `frames/` - Photo frames and similar items

## Usage

Place your room images in the `walls/` folder and your product images in the appropriate `products/` subfolder.
""",
        
        "input/walls/README.md": """# Wall/Room Images

Place room images with visible walls here. 

## Requirements:
- Good lighting and clear wall visibility
- Supported formats: JPG, PNG, BMP, TIFF
- Recommended resolution: 800x600 or higher
- Avoid cluttered walls for better placement results

## Sample Images:
- `living_room_*.jpg` - Living room scenes
- `bedroom_*.jpg` - Bedroom scenes  
- `office_*.jpg` - Office/workspace scenes
""",
        
        "input/products/README.md": """# Product Images

Organize product images by category in the subfolders.

## General Requirements:
- PNG format recommended for transparency support
- Clear product visibility
- Minimal background (transparent preferred)
- Good lighting and contrast
""",
        
        "input/products/tvs/README.md": """# TV Images

Television and monitor images for wall mounting simulation.

## Requirements:
- 16:9 aspect ratio preferred for modern TVs
- PNG with transparent background recommended
- Include bezels and stands if applicable
- Various sizes available (55", 65", etc.)
""",
        
        "input/products/paintings/README.md": """# Painting Images

Artwork, paintings, and decorative images.

## Requirements:
- Include frames in the image
- Various aspect ratios supported
- High quality artwork images
- Consider lighting and shadows in the original image
""",
        
        "input/products/frames/README.md": """# Frame Images

Photo frames and similar wall-mounted items.

## Requirements:
- Portrait or landscape orientations
- Include the frame border
- Family photos, certificates, etc.
- Various frame styles and materials
""",
        
        "output/README.md": """# Output Images

This folder contains processed results from the Wall AR Pipeline.

## Structure:
- `results/` - Final processed images with products placed on walls
- `debug/` - Debug visualizations and intermediate results (when enabled)

## File Naming:
Results are typically named based on the input files or timestamps.
Debug folders contain detailed processing information and visualizations.
"""
    }
    
    for filepath, content in readme_contents.items():
        Path(filepath).write_text(content)
        print(f"✓ Created README: {filepath}")

def main():
    """Main setup function."""
    print("🏗️  Setting up Wall AR Pipeline folder structure...")
    print("=" * 60)
    
    # Create folder structure
    create_folder_structure()
    
    # Move existing samples
    move_existing_samples()
    
    # Create new sample images
    print("\n🎨 Creating sample images...")
    create_sample_wall_images()
    create_sample_product_images()
    
    # Create documentation
    print("\n📝 Creating documentation...")
    create_readme_files()
    
    print("\n" + "=" * 60)
    print("✅ Setup complete!")
    print("\n📁 Folder structure created:")
    print("   input/")
    print("   ├── walls/           (room images)")
    print("   └── products/")
    print("       ├── tvs/         (TV images)")
    print("       ├── paintings/   (artwork images)")
    print("       └── frames/      (photo frames)")
    print("   output/")
    print("   ├── results/         (final outputs)")
    print("   └── debug/           (debug info)")
    print("\n🎯 Usage examples:")
    print("   python wall_ar_pipeline.py --room input/walls/living_room_1.jpg --product input/products/tvs/modern_tv_55inch.png --type tv --output output/results/result.png")
    print("   python wall_ar_pipeline.py --room input/walls/bedroom_1.jpg --product input/products/paintings/abstract_art.png --type painting --output output/results/bedroom_art.png")

if __name__ == "__main__":
    main()
