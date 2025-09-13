"""
Configuration settings for the Wall Segmentation & Product Placement system.
"""

import os
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class ModelConfig:
    """Configuration for AI models."""
    sam_model_type: str = "vit_h"  # vit_h, vit_l, vit_b
    sam_checkpoint_path: Optional[str] = None
    device: str = "cuda" if os.getenv("CUDA_AVAILABLE", "false").lower() == "true" else "cpu"
    
@dataclass
class ProcessingConfig:
    """Configuration for image processing parameters."""
    max_image_size: Tuple[int, int] = (1920, 1080)
    wall_confidence_threshold: float = 0.4
    blend_alpha: float = 0.8
    lighting_adjustment_strength: float = 0.3
    
@dataclass
class PlacementConfig:
    """Configuration for product placement parameters."""
    default_tv_size_ratio: float = 0.15  # TV size as ratio of wall width
    default_painting_size_ratio: float = 0.12  # Painting size as ratio of wall width
    min_wall_distance: int = 50  # Minimum distance from wall edges in pixels
    perspective_correction_enabled: bool = True
    
@dataclass
class OutputConfig:
    """Configuration for output settings."""
    output_format: str = "PNG"
    output_quality: int = 95
    save_intermediate_results: bool = False
    debug_mode: bool = False

# Global configuration instance
config = {
    "model": ModelConfig(),
    "processing": ProcessingConfig(),
    "placement": PlacementConfig(),
    "output": OutputConfig()
}
