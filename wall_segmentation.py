"""
Wall Segmentation Module

This module provides functionality to detect and segment wall regions from room images
using various AI vision models including SAM (Segment Anything Model) and custom approaches.
"""

import cv2
import numpy as np
import torch
from typing import Tuple, Optional, List
from PIL import Image
import logging
from dataclasses import dataclass

try:
    from unet_segmentation import UNetWallSegmenter
    UNET_AVAILABLE = True
except ImportError:
    UNET_AVAILABLE = False
    logging.warning("U-Net model not available. Using traditional methods only.")

@dataclass
class WallSegmentationResult:
    """Result container for wall segmentation."""
    mask: np.ndarray
    confidence: float
    wall_corners: List[Tuple[int, int]]
    dominant_plane: Optional[np.ndarray] = None

class WallSegmentationError(Exception):
    """Custom exception for wall segmentation errors."""
    pass

class WallSegmenter:
    """
    Main class for wall segmentation using U-Net and traditional approaches.
    
    Supports:
    - U-Net for semantic segmentation of walls
    - Traditional computer vision methods as fallback
    - Hybrid approach combining multiple techniques
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        """
        Initialize the wall segmenter.
        
        Args:
            model_path: Path to U-Net model checkpoint (optional)
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.device = device
        self.unet_segmenter = None
        
        if UNET_AVAILABLE:
            try:
                self.unet_segmenter = UNetWallSegmenter(model_path=model_path, device=device)
                logging.info("U-Net segmenter initialized successfully")
            except Exception as e:
                logging.warning(f"Failed to initialize U-Net: {e}. Using traditional methods only.")
        else:
            logging.info("Using traditional CV methods only")
    
    def segment_wall(self, image: np.ndarray, method: str = "auto") -> WallSegmentationResult:
        """
        Segment wall from room image.
        
        Args:
            image: Input room image as numpy array (H, W, 3)
            method: Segmentation method ('unet', 'traditional', 'auto')
            
        Returns:
            WallSegmentationResult containing mask and metadata
        """
        if method == "auto":
            method = "unet" if self.unet_segmenter else "traditional"
        
        if method == "unet" and self.unet_segmenter:
            return self.unet_segmenter.segment_wall(image, method="unet")
        else:
            return self._segment_traditional(image)
    
    
    def _segment_traditional(self, image: np.ndarray) -> WallSegmentationResult:
        """Segment wall using traditional computer vision methods."""
        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Edge detection for structural elements
        edges = cv2.Canny(gray, 50, 150)
        
        # Line detection for wall boundaries
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                               minLineLength=100, maxLineGap=10)
        
        # Create wall mask using multiple cues
        wall_mask = self._create_wall_mask_traditional(image, gray, hsv, lines)
        
        # Calculate confidence based on mask properties
        confidence = self._calculate_mask_confidence(wall_mask, image.shape[:2])
        
        # Extract wall corners
        wall_corners = self._extract_wall_corners(wall_mask)
        
        return WallSegmentationResult(
            mask=wall_mask,
            confidence=confidence,
            wall_corners=wall_corners
        )
    
    
    def _create_wall_mask_traditional(self, image: np.ndarray, gray: np.ndarray, 
                                    hsv: np.ndarray, lines: Optional[np.ndarray]) -> np.ndarray:
        """Create wall mask using traditional computer vision techniques."""
        h, w = gray.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Method 1: Color-based segmentation
        # Assume walls are often neutral colors
        saturation = hsv[:, :, 1]
        low_saturation = saturation < 50  # Low saturation regions
        
        # Method 2: Texture analysis
        # Walls typically have uniform texture
        blur = cv2.GaussianBlur(gray, (15, 15), 0)
        texture_variance = cv2.absdiff(gray, blur)
        low_texture = texture_variance < 20
        
        # Method 3: Geometric constraints
        # Walls are typically large, connected regions
        combined = (low_saturation & low_texture).astype(np.uint8)
        
        # Morphological operations to clean up
        kernel = np.ones((7, 7), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        
        # Find largest connected component (likely the main wall)
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.fillPoly(mask, [largest_contour], 255)
        
        return mask
    
    def _extract_wall_corners(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        """Extract corner points of the wall from the segmentation mask."""
        # Find contours of the mask
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
        
        # Get the largest contour (main wall)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Extract corner points
        corners = [(int(point[0][0]), int(point[0][1])) for point in approx]
        
        return corners
    
    def _calculate_mask_confidence(self, mask: np.ndarray, image_shape: Tuple[int, int]) -> float:
        """Calculate confidence score for the segmentation mask."""
        h, w = image_shape
        total_pixels = h * w
        mask_pixels = np.sum(mask > 0)
        
        # Confidence based on mask size (walls should be reasonably large)
        size_ratio = mask_pixels / total_pixels
        
        if size_ratio < 0.1 or size_ratio > 0.8:  # Too small or too large
            size_confidence = 0.3
        else:
            size_confidence = 1.0
        
        # Confidence based on mask shape (walls should be somewhat rectangular)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            if perimeter > 0:
                compactness = 4 * np.pi * area / (perimeter ** 2)
                shape_confidence = min(compactness * 2, 1.0)  # Normalize
            else:
                shape_confidence = 0.3
        else:
            shape_confidence = 0.3
        
        return (size_confidence + shape_confidence) / 2

    def visualize_segmentation(self, image: np.ndarray, result: WallSegmentationResult) -> np.ndarray:
        """
        Visualize segmentation result with overlay.
        
        Args:
            image: Original image
            result: Segmentation result
            
        Returns:
            Visualization image with mask overlay and corner points
        """
        # Use U-Net visualizer if available, otherwise use traditional
        if self.unet_segmenter:
            return self.unet_segmenter.visualize_segmentation(image, result)
        
        # Traditional visualization fallback
        # Ensure input image is uint8
        if image.dtype != np.uint8:
            vis_image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        else:
            vis_image = image.copy()
        
        # Ensure mask is boolean or uint8
        mask = result.mask
        if mask.dtype == bool:
            mask_positions = mask
        else:
            mask_positions = mask > 0
        
        # Create colored mask overlay with same dtype as vis_image
        mask_colored = np.zeros_like(vis_image, dtype=np.uint8)
        if len(mask_colored.shape) == 3:  # Color image
            mask_colored[mask_positions, 0] = 0    # Blue channel
            mask_colored[mask_positions, 1] = 255  # Green channel  
            mask_colored[mask_positions, 2] = 0    # Red channel
        else:  # Grayscale
            mask_colored[mask_positions] = 255
        
        # Ensure both images are uint8 before blending
        vis_image = vis_image.astype(np.uint8)
        mask_colored = mask_colored.astype(np.uint8)
        
        # Blend with original image
        vis_image = cv2.addWeighted(vis_image, 0.7, mask_colored, 0.3, 0)
        
        # Draw corner points
        for corner in result.wall_corners:
            cv2.circle(vis_image, corner, 5, (255, 0, 0), -1)
        
        # Add confidence text
        cv2.putText(vis_image, f"Traditional CV Confidence: {result.confidence:.2f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return vis_image
