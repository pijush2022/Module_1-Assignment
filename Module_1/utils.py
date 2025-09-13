"""
Utility Functions

This module provides helper functions and utilities used across the wall segmentation
and product placement system.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Union
from PIL import Image, ImageEnhance
import logging
from pathlib import Path
import json

def validate_image_format(image_path: Union[str, Path]) -> bool:
    """
    Validate if the image format is supported.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        True if format is supported, False otherwise
    """
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    path = Path(image_path)
    return path.suffix.lower() in supported_formats

def ensure_rgb_format(image: np.ndarray) -> np.ndarray:
    """
    Ensure image is in RGB format.
    
    Args:
        image: Input image array
        
    Returns:
        Image in RGB format
    """
    if len(image.shape) == 2:
        # Grayscale to RGB
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        # RGBA to RGB (remove alpha channel)
        return image[:, :, :3]
    elif image.shape[2] == 3:
        # Already RGB
        return image
    else:
        raise ValueError(f"Unsupported image format with shape: {image.shape}")

def calculate_image_quality_score(image: np.ndarray) -> float:
    """
    Calculate a quality score for the input image.
    
    Args:
        image: Input image array
        
    Returns:
        Quality score between 0 and 1
    """
    # Convert to grayscale for analysis
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Calculate various quality metrics
    
    # 1. Sharpness (Laplacian variance)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_score = min(laplacian_var / 1000, 1.0)  # Normalize
    
    # 2. Contrast (standard deviation)
    contrast_score = min(np.std(gray) / 64, 1.0)  # Normalize
    
    # 3. Brightness distribution
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_normalized = hist / hist.sum()
    
    # Avoid images that are too dark or too bright
    dark_pixels = np.sum(hist_normalized[:50])
    bright_pixels = np.sum(hist_normalized[200:])
    
    brightness_score = 1.0 - max(dark_pixels, bright_pixels)
    
    # 4. Overall quality (weighted average)
    quality_score = (
        0.4 * sharpness_score +
        0.3 * contrast_score +
        0.3 * brightness_score
    )
    
    return quality_score

def create_thumbnail(image: np.ndarray, size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """
    Create a thumbnail of the image while maintaining aspect ratio.
    
    Args:
        image: Input image array
        size: Target thumbnail size (width, height)
        
    Returns:
        Thumbnail image
    """
    h, w = image.shape[:2]
    target_w, target_h = size
    
    # Calculate scale to fit within target size
    scale = min(target_w / w, target_h / h)
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    thumbnail = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create canvas with target size
    canvas = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
    
    # Center the thumbnail on canvas
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = thumbnail
    
    return canvas

def enhance_image_quality(image: np.ndarray, 
                         brightness: float = 1.0,
                         contrast: float = 1.0,
                         saturation: float = 1.0) -> np.ndarray:
    """
    Enhance image quality with brightness, contrast, and saturation adjustments.
    
    Args:
        image: Input image array
        brightness: Brightness factor (1.0 = no change)
        contrast: Contrast factor (1.0 = no change)
        saturation: Saturation factor (1.0 = no change)
        
    Returns:
        Enhanced image
    """
    # Convert to PIL for enhancement
    pil_image = Image.fromarray(image)
    
    # Apply brightness
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(brightness)
    
    # Apply contrast
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(contrast)
    
    # Apply saturation
    if saturation != 1.0:
        enhancer = ImageEnhance.Color(pil_image)
        pil_image = enhancer.enhance(saturation)
    
    return np.array(pil_image)

def detect_room_type(image: np.ndarray) -> str:
    """
    Detect the type of room from the image (basic heuristics).
    
    Args:
        image: Input room image
        
    Returns:
        Detected room type ('living_room', 'bedroom', 'office', 'unknown')
    """
    # This is a simplified room detection based on color analysis
    # In a real implementation, you might use a trained classifier
    
    # Convert to HSV for color analysis
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Analyze dominant colors
    hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256])
    
    # Simple heuristics based on color distribution
    warm_colors = np.sum(hist_h[0:30]) + np.sum(hist_h[150:180])  # Reds/oranges
    cool_colors = np.sum(hist_h[90:150])  # Blues/greens
    
    avg_saturation = np.average(np.arange(256), weights=hist_s.flatten())
    avg_brightness = np.average(np.arange(256), weights=hist_v.flatten())
    
    # Basic classification rules
    if warm_colors > cool_colors and avg_saturation > 100:
        return "living_room"
    elif avg_brightness < 100 and avg_saturation < 80:
        return "bedroom"
    elif cool_colors > warm_colors and avg_brightness > 150:
        return "office"
    else:
        return "unknown"

def calculate_wall_area_ratio(wall_mask: np.ndarray, image_shape: Tuple[int, int]) -> float:
    """
    Calculate the ratio of wall area to total image area.
    
    Args:
        wall_mask: Binary mask of wall region
        image_shape: Shape of the original image (height, width)
        
    Returns:
        Wall area ratio (0.0 to 1.0)
    """
    total_pixels = image_shape[0] * image_shape[1]
    wall_pixels = np.sum(wall_mask > 0)
    
    return wall_pixels / total_pixels if total_pixels > 0 else 0.0

def find_optimal_tv_size(wall_corners: List[Tuple[int, int]], 
                        room_type: str = "living_room") -> Tuple[int, int]:
    """
    Calculate optimal TV size based on wall dimensions and room type.
    
    Args:
        wall_corners: Corner points of the wall
        room_type: Type of room
        
    Returns:
        Optimal TV dimensions (width, height)
    """
    # Calculate wall dimensions
    if len(wall_corners) < 2:
        return (400, 225)  # Default 16:9 TV
    
    corners = np.array(wall_corners)
    
    # Find wall width (maximum distance between corners)
    max_distance = 0
    for i in range(len(corners)):
        for j in range(i + 1, len(corners)):
            distance = np.linalg.norm(corners[i] - corners[j])
            max_distance = max(max_distance, distance)
    
    wall_width = max_distance
    
    # Room-specific size ratios
    size_ratios = {
        "living_room": 0.15,
        "bedroom": 0.12,
        "office": 0.10,
        "unknown": 0.12
    }
    
    ratio = size_ratios.get(room_type, 0.12)
    tv_width = int(wall_width * ratio)
    tv_height = int(tv_width / (16/9))  # 16:9 aspect ratio
    
    return (tv_width, tv_height)

def create_product_mask(product_image: np.ndarray, 
                       threshold: int = 30) -> np.ndarray:
    """
    Create a mask for the product image to separate it from background.
    
    Args:
        product_image: Product image (RGB or RGBA)
        threshold: Threshold for background detection
        
    Returns:
        Binary mask of the product
    """
    if product_image.shape[2] == 4:
        # Use alpha channel if available
        alpha = product_image[:, :, 3]
        mask = (alpha > threshold).astype(np.uint8) * 255
    else:
        # Create mask based on non-black pixels
        gray = cv2.cvtColor(product_image, cv2.COLOR_RGB2GRAY)
        mask = (gray > threshold).astype(np.uint8) * 255
    
    # Clean up mask with morphological operations
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask

def save_processing_report(results: dict, output_path: Union[str, Path]):
    """
    Save a detailed processing report.
    
    Args:
        results: Dictionary containing processing results and metadata
        output_path: Path to save the report
    """
    report = {
        "timestamp": str(np.datetime64('now')),
        "processing_summary": {
            "total_time": results.get("processing_time", 0),
            "confidence_score": results.get("confidence_score", 0),
            "success": results.get("success", False)
        },
        "segmentation_details": {
            "method_used": results.get("segmentation_method", "unknown"),
            "wall_confidence": results.get("wall_confidence", 0),
            "wall_area_ratio": results.get("wall_area_ratio", 0)
        },
        "placement_details": {
            "product_type": results.get("product_type", "unknown"),
            "placement_position": results.get("placement_position", [0, 0]),
            "product_size": results.get("product_size", [0, 0]),
            "collision_detected": results.get("collision_detected", False)
        },
        "quality_metrics": {
            "input_image_quality": results.get("input_quality", 0),
            "output_image_quality": results.get("output_quality", 0),
            "lighting_match_applied": results.get("lighting_adjusted", False),
            "shadows_added": results.get("shadows_added", False)
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

def load_config_from_file(config_path: Union[str, Path]) -> dict:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return json.load(f)

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Optional log file path
    """
    level = getattr(logging, log_level.upper())
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

def benchmark_processing_time(func):
    """
    Decorator to benchmark processing time of functions.
    
    Args:
        func: Function to benchmark
        
    Returns:
        Wrapped function with timing
    """
    import time
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger = logging.getLogger(func.__module__)
        logger.info(f"{func.__name__} took {end_time - start_time:.3f} seconds")
        
        return result
    
    return wrapper
