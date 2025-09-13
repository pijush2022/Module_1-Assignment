"""
Installation script for real Stable Diffusion dependencies
Handles potential xformers issues gracefully
"""

import subprocess
import sys
import os

def install_package(package, fallback=None):
    """Install package with fallback option"""
    try:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ {package} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install {package}: {e}")
        if fallback:
            print(f"Trying fallback: {fallback}")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", fallback])
                print(f"✓ {fallback} installed successfully")
                return True
            except subprocess.CalledProcessError:
                print(f"✗ Fallback {fallback} also failed")
        return False

def main():
    """Install Stable Diffusion dependencies"""
    print("=== Installing Real Stable Diffusion Dependencies ===")
    
    # Core packages (required)
    core_packages = [
        "torch>=2.0.0",
        "torchvision>=0.15.0", 
        "diffusers>=0.21.0",
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "safetensors>=0.3.0",
        "huggingface-hub>=0.15.0"
    ]
    
    # Optional packages (may fail on some systems)
    optional_packages = [
        ("controlnet-aux>=0.0.6", "opencv-python>=4.7.0"),  # Fallback to opencv
        ("xformers>=0.0.20", None)  # Skip if fails
    ]
    
    # Image processing
    image_packages = [
        "Pillow>=9.5.0",
        "opencv-python>=4.7.0",
        "numpy>=1.24.0",
        "scikit-image>=0.20.0",
        "matplotlib>=3.7.0"
    ]
    
    print("\n1. Installing core packages...")
    for package in core_packages:
        install_package(package)
    
    print("\n2. Installing image processing packages...")
    for package in image_packages:
        install_package(package)
    
    print("\n3. Installing optional packages...")
    for package, fallback in optional_packages:
        if not install_package(package, fallback):
            print(f"Skipping {package} - not critical for basic functionality")
    
    print("\n=== Installation Complete ===")
    print("You can now run: python stable_diffusion_ar.py")
    print("\nIf you encounter issues:")
    print("- Run: python product_placement_simple.py (concept demo)")
    print("- Check CUDA availability: python -c 'import torch; print(torch.cuda.is_available())'")

if __name__ == "__main__":
    main()
