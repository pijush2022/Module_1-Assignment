# AI-Powered AR Product Placement System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Last Updated](https://img.shields.io/badge/Last%20Updated-September%202025-green.svg)](#)

A comprehensive AI system for realistic product placement in room environments, featuring two complementary approaches: traditional computer vision with U-Net segmentation and advanced Stable Diffusion with ControlNet conditioning.

## 🎯 Project Overview

This repository contains two specialized modules that demonstrate different approaches to AI-powered product placement for AR applications:

- **Module 1**: Wall AR System with AI-powered wall segmentation and intelligent product placement
- **Module 2**: Stable Diffusion pipeline with ControlNet for photorealistic product generation

Both modules are designed to create stunning AR try-on experiences by seamlessly placing TVs, paintings, and frames on walls with photorealistic perspective, lighting, and shadows.

## 🏗️ Repository Structure

```
AI-AR-Product-Placement/
├── Module_1/                          # Wall AR System - Traditional CV + U-Net
│   ├── 🔧 Core System
│   │   ├── config.py                  # Centralized configuration
│   │   ├── wall_segmentation.py       # Multi-model wall detection
│   │   ├── unet_segmentation.py       # U-Net implementation
│   │   ├── product_placement.py       # Smart positioning engine
│   │   ├── image_blending.py          # Photorealistic compositing
│   │   └── wall_ar_pipeline.py        # Main orchestrator
│   │
│   ├── 📂 Data Structure
│   │   ├── input/                     # Source images
│   │   │   ├── walls/                 # Room photographs
│   │   │   └── products/              # Product catalog
│   │   ├── output/                    # Generated results
│   │   └── models/                    # AI model storage
│   │
│   └── 🧪 Testing & Examples
│       ├── example_usage.py           # Comprehensive demos
│       ├── quick_test_with_unet.py    # U-Net validation
│       └── setup_folders.py           # Project initialization
│
├── Module_2/                          # Stable Diffusion AR System
│   ├── 🎨 Core Pipeline
│   │   ├── product_placement.py       # Main ControlNet pipeline
│   │   ├── product_placement_cpu.py   # CPU-optimized version
│   │   └── stable_diffusion_ar.py     # Full SD implementation
│   │
│   ├── 🔬 Fine-tuning Framework
│   │   ├── fine_tuning_demo.py        # DreamBooth & LoRA setup
│   │   └── evaluation_metrics.py      # Quality assessment
│   │
│   └── 📊 Configuration
│       ├── config.py                  # SD-specific settings
│       └── requirements.txt           # Dependencies
│
└── README.md                          # This file
```

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+** (3.9+ recommended)
- **8GB+ RAM** (16GB+ for Stable Diffusion)
- **GPU optional** (CUDA-compatible for acceleration)

### Option 1: Traditional CV + U-Net Approach (Module 1)
```bash
# Navigate to Module 1
cd Module_1

# Install dependencies
pip install -r requirements.txt

# Initialize project structure
python setup_folders.py

# Run quick test with U-Net
python quick_test_with_unet.py
```

### Option 2: Stable Diffusion Approach (Module 2)
```bash
# Navigate to Module 2
cd Module_2

# Install dependencies
pip install -r requirements.txt

# Run simple demo (no ML dependencies)
python product_placement_simple.py

# OR run full pipeline (requires GPU)
python product_placement.py
```

## 🎯 Module Comparison

| Feature | Module 1 (U-Net + CV) | Module 2 (Stable Diffusion) |
|---------|----------------------|----------------------------|
| **Speed** | ⚡ 3-8 seconds | 🐌 15-60 seconds |
| **Quality** | ⭐⭐⭐⭐ High | ⭐⭐⭐⭐⭐ Photorealistic |
| **Memory** | 💾 1-2GB | 💾 4-8GB |
| **Setup** | 🟢 Easy | 🟡 Complex |
| **Real-time** | ✅ Yes | ❌ No |
| **Customization** | 🔧 Good | 🎨 Excellent |
| **Best For** | Production apps | Research & demos |

## 🎨 Key Features

### Module 1: Wall AR System
- **🧠 Advanced Wall Detection**: U-Net architecture with SAM and traditional CV fallbacks
- **📐 Smart Product Placement**: Intelligent positioning with automatic perspective correction
- **✨ Photorealistic Blending**: Advanced lighting analysis, shadow generation, and seamless integration
- **⚡ Performance Optimized**: Multiple model options for different speed/quality requirements
- **📱 Mobile Friendly**: Lightweight architecture suitable for real-time applications

### Module 2: Stable Diffusion AR
- **🎯 ControlNet Integration**: Depth-based spatial conditioning for realistic placement
- **📏 Size Variations**: Demonstrates multiple product sizes (42", 55", 65" TVs)
- **🔬 Fine-tuning Framework**: DreamBooth/LoRA configurations for custom training
- **📊 Comprehensive Evaluation**: CLIP scores, LPIPS, depth consistency metrics
- **🎨 Photorealistic Generation**: State-of-the-art diffusion model quality

## 🛠️ Installation Guide

### Module 1 Setup
```bash
cd Module_1

# Create virtual environment
python -m venv wall_ar_env
wall_ar_env\Scripts\activate  # Windows
# source wall_ar_env/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Verify installation
python quick_test_with_unet.py
```

### Module 2 Setup
```bash
cd Module_2

# Create virtual environment
python -m venv sd_ar_env
sd_ar_env\Scripts\activate  # Windows

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt

# Run demo
python product_placement_simple.py
```

## 🎮 Usage Examples

### Module 1: Quick Product Placement
```python
from Module_1.wall_ar_pipeline import WallARPipeline
from Module_1.product_placement import ProductType

# Initialize with U-Net (recommended)
pipeline = WallARPipeline()

# Place a TV on living room wall
result = pipeline.process_image(
    room_image_path="Module_1/input/walls/living_room.jpg",
    product_image_path="Module_1/input/products/tvs/modern_tv.png", 
    product_type=ProductType.TV,
    output_path="Module_1/output/results/living_room_with_tv.png"
)

print(f"✅ Success! Confidence: {result.confidence_score:.2f}")
```

### Module 2: Stable Diffusion Placement
```python
from Module_2.product_placement_simple import SimpleProductPlacementDemo

# Initialize demo
demo = SimpleProductPlacementDemo()

# Run TV placement demonstration
demo.demonstrate_tv_placement()

# Run size comparison (42" vs 55" vs 65")
demo.demonstrate_size_comparison()
```

## 📊 Performance Benchmarks

### Module 1 (U-Net + CV)
- **Processing Time**: 3-8 seconds (CPU), 1-3 seconds (GPU)
- **Memory Usage**: 1-2GB RAM
- **Quality Score**: 4/5 stars
- **Best For**: Real-time applications, mobile apps

### Module 2 (Stable Diffusion)
- **Processing Time**: 15-60 seconds (depending on steps)
- **Memory Usage**: 4-8GB VRAM
- **Quality Score**: 5/5 stars
- **Best For**: High-quality demonstrations, research

## 🎯 Product Types Supported

Both modules support:
- **📺 TVs**: Various sizes (32", 42", 55", 65", 75")
- **🖼️ Paintings**: Small, medium, large with frame effects
- **🖼️ Photo Frames**: Portrait and landscape orientations
- **🪞 Mirrors**: Various shapes and sizes

## 🔧 Configuration

### Module 1 Configuration
```python
from Module_1.config import config

# Customize processing settings
config["processing"].max_image_size = (1920, 1080)
config["placement"].default_tv_size_ratio = 0.18
config["output"].debug_mode = True
```

### Module 2 Configuration
```python
from Module_2.config import Config

# Modify generation parameters
Config.DEFAULT_STEPS = 30
Config.DEFAULT_GUIDANCE_SCALE = 8.0
Config.IMAGE_RESOLUTION = 512
```

## 🧪 Testing & Validation

### Module 1 Tests
```bash
cd Module_1

# Quick validation
python quick_test_with_unet.py

# Comprehensive examples
python example_usage.py

# Custom image testing
python wall_ar_pipeline.py --room your_room.jpg --product your_tv.png --type tv
```

### Module 2 Tests
```bash
cd Module_2

# Simple demo (no GPU required)
python product_placement_simple.py

# Full pipeline test
python product_placement.py

# Evaluation metrics
python evaluation_metrics.py
```

## 📈 Evaluation Metrics

### Module 1 Quality Assessment
- **Segmentation Confidence**: Wall detection accuracy (0-1)
- **Placement Confidence**: Position assessment (0-1)
- **Overall Confidence**: Weighted combination (0-1)

### Module 2 Quality Assessment
- **CLIP Score**: Text-image semantic alignment
- **LPIPS Score**: Perceptual similarity
- **Depth Consistency**: Spatial understanding
- **Scale Accuracy**: Correct product sizing

## 🤝 Contributing

1. Fork the repository
2. Choose your module: `cd Module_1` or `cd Module_2`
3. Create feature branch: `git checkout -b feature-name`
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit pull request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

### Module 1
- PyTorch team for U-Net implementation
- OpenCV community for computer vision tools
- SAM (Segment Anything Model) by Meta AI

### Module 2
- Hugging Face Diffusers team
- Stability AI for Stable Diffusion models
- ControlNet authors for spatial conditioning
- MiDaS team for depth estimation

## 📞 Support

- **Issues**: GitHub Issues page
- **Module 1 Documentation**: See `Module_1/README.md`
- **Module 2 Documentation**: See `Module_2/README.md`
- **Examples**: Check respective module demo files

---

**Choose Your Approach:**
- **Need real-time performance?** → Use Module 1 (U-Net + CV)
- **Want maximum quality?** → Use Module 2 (Stable Diffusion)
- **Building production app?** → Start with Module 1
- **Research or demos?** → Explore Module 2

Both modules can be used independently or together for comprehensive AR product placement solutions.
