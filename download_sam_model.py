"""
Script to download SAM ViT-H model checkpoint with progress reporting and file verification.
"""

import requests
from pathlib import Path
import os

def download_sam_model():
    """Download SAM ViT-H model checkpoint."""
    
    # Model URL and details
    model_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    model_path = Path("models/sam_vit_h_4b8939.pth")
    expected_size = 2564550879  # ~2.6GB
    
    # Create models directory
    model_path.parent.mkdir(exist_ok=True)
    
    # Check if model already exists
    if model_path.exists():
        current_size = model_path.stat().st_size
        if current_size == expected_size:
            print(f"✅ SAM model already exists and is complete ({current_size / (1024**3):.2f} GB)")
            return True
        else:
            print(f"⚠️ SAM model exists but incomplete ({current_size / (1024**3):.2f} GB)")
            print("🔄 Re-downloading...")
    
    print(f"📥 Downloading SAM ViT-H model...")
    print(f"   URL: {model_url}")
    print(f"   Size: ~{expected_size / (1024**3):.2f} GB")
    
    try:
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Progress reporting
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\r📊 Progress: {progress:.1f}% ({downloaded / (1024**2):.1f} MB)", end='')
        
        print()  # New line after progress
        
        # Verify download
        final_size = model_path.stat().st_size
        if final_size == expected_size:
            print(f"✅ Download successful! ({final_size / (1024**3):.2f} GB)")
            return True
        else:
            print(f"❌ Download incomplete. Expected {expected_size}, got {final_size}")
            return False
            
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

def verify_model():
    """Verify SAM model file integrity."""
    model_path = Path("models/sam_vit_h_4b8939.pth")
    
    if not model_path.exists():
        print("❌ SAM model file not found")
        return False
    
    file_size = model_path.stat().st_size
    expected_size = 2564550879
    
    if file_size == expected_size:
        print(f"✅ SAM model verified ({file_size / (1024**3):.2f} GB)")
        return True
    else:
        print(f"❌ SAM model size mismatch. Expected {expected_size}, got {file_size}")
        return False

if __name__ == "__main__":
    print("🤖 SAM Model Downloader")
    print("=" * 40)
    
    if verify_model():
        print("Model already available and verified.")
    else:
        success = download_sam_model()
        if success:
            verify_model()
        else:
            print("Download failed. Please try again or download manually.")
