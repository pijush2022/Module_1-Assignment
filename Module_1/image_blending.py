"""
Image Blending and Lighting Adjustment Module

This module provides advanced blending techniques and lighting adjustments
to create realistic integration of products into room scenes.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
from scipy import ndimage
from skimage import exposure, filters
import logging
from dataclasses import dataclass

@dataclass
class LightingAnalysis:
    """Container for lighting analysis results."""
    dominant_light_direction: Tuple[float, float]
    ambient_intensity: float
    shadow_regions: np.ndarray
    highlight_regions: np.ndarray
    color_temperature: float

@dataclass
class BlendingResult:
    """Container for blending operation results."""
    blended_image: np.ndarray
    blend_mask: np.ndarray
    lighting_adjusted: bool
    shadow_added: bool

class ImageBlender:
    """
    Advanced image blending system with lighting-aware compositing.
    
    Features:
    - Poisson blending for seamless integration
    - Lighting direction analysis and matching
    - Shadow generation and placement
    - Color temperature adjustment
    - Edge feathering and anti-aliasing
    """
    
    def __init__(self):
        """Initialize the image blender."""
        self.logger = logging.getLogger(__name__)
    
    def blend_with_lighting(self, 
                           background: np.ndarray,
                           foreground: np.ndarray,
                           position: Tuple[int, int],
                           wall_mask: np.ndarray,
                           enable_shadows: bool = True,
                           enable_lighting_match: bool = True) -> BlendingResult:
        """
        Blend foreground into background with advanced lighting adjustments.
        
        Args:
            background: Background room image
            foreground: Product image to blend (RGB or RGBA)
            position: Position to place foreground (x, y)
            wall_mask: Mask of the wall region
            enable_shadows: Whether to generate realistic shadows
            enable_lighting_match: Whether to match lighting conditions
            
        Returns:
            BlendingResult with the final composite and metadata
        """
        # Analyze lighting in the background
        lighting_analysis = self._analyze_lighting(background, wall_mask)
        
        # Prepare foreground image
        fg_prepared = self._prepare_foreground(foreground, lighting_analysis if enable_lighting_match else None)
        
        # Create blend mask with feathered edges
        blend_mask = self._create_blend_mask(fg_prepared, feather_radius=5)
        
        # Perform initial blending
        initial_blend = self._poisson_blend(background, fg_prepared, position, blend_mask)
        
        # Add shadows if enabled
        if enable_shadows:
            shadow_blend = self._add_realistic_shadows(
                initial_blend, fg_prepared, position, lighting_analysis, wall_mask
            )
        else:
            shadow_blend = initial_blend
        
        # Add ambient lighting effects
        if enable_lighting_match:
            shadow_blend = self._add_ambient_lighting(
                shadow_blend, lighting_analysis
            )
        
        # Final color and lighting adjustments
        final_image = self._apply_final_adjustments(
            shadow_blend, background, position, fg_prepared.shape[:2]
        )
        
        return BlendingResult(
            blended_image=final_image,
            blend_mask=blend_mask,
            lighting_adjusted=enable_lighting_match,
            shadow_added=enable_shadows
        )
    
    def _analyze_lighting(self, image: np.ndarray, wall_mask: np.ndarray) -> LightingAnalysis:
        """Analyze lighting conditions in the room image."""
        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Focus analysis on wall region
        wall_region = gray * (wall_mask / 255.0)
        
        # Analyze light direction using gradient
        grad_x = cv2.Sobel(wall_region, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(wall_region, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate dominant gradient direction (light direction)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_angle = np.arctan2(grad_y, grad_x)
        
        # Weight by magnitude and calculate dominant direction
        valid_gradients = gradient_magnitude > np.percentile(gradient_magnitude, 70)
        if np.any(valid_gradients):
            weighted_angles = gradient_angle[valid_gradients]
            dominant_angle = np.median(weighted_angles)
            light_direction = (np.cos(dominant_angle), np.sin(dominant_angle))
        else:
            light_direction = (-0.5, -0.5)  # Default: top-left lighting
        
        # Calculate ambient intensity
        wall_pixels = wall_region[wall_mask > 0]
        ambient_intensity = np.mean(wall_pixels) if len(wall_pixels) > 0 else 128
        
        # Identify shadow and highlight regions
        shadow_threshold = ambient_intensity * 0.7
        highlight_threshold = ambient_intensity * 1.3
        
        shadow_regions = (wall_region < shadow_threshold) & (wall_mask > 0)
        highlight_regions = (wall_region > highlight_threshold) & (wall_mask > 0)
        
        # Estimate color temperature from wall region
        wall_rgb = image[wall_mask > 0]
        if len(wall_rgb) > 0:
            avg_color = np.mean(wall_rgb, axis=0)
            # Simple color temperature estimation based on R/B ratio
            color_temp = self._estimate_color_temperature(avg_color)
        else:
            color_temp = 5500  # Default daylight
        
        return LightingAnalysis(
            dominant_light_direction=light_direction,
            ambient_intensity=ambient_intensity / 255.0,
            shadow_regions=shadow_regions.astype(np.uint8),
            highlight_regions=highlight_regions.astype(np.uint8),
            color_temperature=color_temp
        )
    
    def _estimate_color_temperature(self, rgb_color: np.ndarray) -> float:
        """Estimate color temperature from RGB values."""
        r, g, b = rgb_color
        
        # Avoid division by zero
        if b == 0:
            b = 1
        
        # Simple estimation based on R/B ratio
        rb_ratio = r / b
        
        # Map ratio to temperature (rough approximation)
        if rb_ratio > 1.5:
            return 3000  # Warm light
        elif rb_ratio > 1.2:
            return 4000  # Slightly warm
        elif rb_ratio > 0.9:
            return 5500  # Daylight
        else:
            return 7000  # Cool light
    
    def _prepare_foreground(self, foreground: np.ndarray, 
                           lighting_analysis: Optional[LightingAnalysis]) -> np.ndarray:
        """Prepare foreground image with lighting adjustments."""
        if foreground.shape[2] == 4:
            # Convert RGBA to RGB
            fg_rgb = foreground[:, :, :3]
            alpha = foreground[:, :, 3]
            
            # Apply alpha to RGB
            fg_prepared = fg_rgb.copy()
            for c in range(3):
                fg_prepared[:, :, c] = fg_rgb[:, :, c] * (alpha / 255.0)
        else:
            fg_prepared = foreground.copy()
        
        if lighting_analysis is not None:
            fg_prepared = self._adjust_lighting_match(foreground, lighting_analysis)
        
        return fg_prepared
    
    def _adjust_lighting_match(self, foreground: np.ndarray, lighting_analysis: LightingAnalysis) -> np.ndarray:
        """Adjust foreground lighting to match background."""
        fg_prepared = foreground.copy()
        
        # More aggressive brightness matching for realism
        brightness_factor = max(0.6, min(1.4, lighting_analysis.ambient_intensity * 1.2))
        fg_prepared = cv2.convertScaleAbs(fg_prepared, alpha=brightness_factor, beta=0)
        
        # Adjust color temperature more noticeably
        fg_prepared = self._adjust_color_temperature(fg_prepared, lighting_analysis.color_temperature)
        
        # Add subtle ambient lighting effect
        fg_prepared = self._add_ambient_lighting(fg_prepared, lighting_analysis)
        
        return fg_prepared
    
    def _adjust_color_temperature(self, image: np.ndarray, target_temp: float) -> np.ndarray:
        """Adjust image color temperature."""
        # Convert to float for processing
        img_float = image.astype(np.float32) / 255.0
        
        # Simple color temperature adjustment
        if target_temp < 4000:  # Warm
            # Increase red, decrease blue
            img_float[:, :, 0] *= 1.1  # More red
            img_float[:, :, 2] *= 0.9  # Less blue
        elif target_temp > 6000:  # Cool
            # Decrease red, increase blue
            img_float[:, :, 0] *= 0.9  # Less red
            img_float[:, :, 2] *= 1.1  # More blue
        
        # Convert back to uint8
        return np.clip(img_float * 255, 0, 255).astype(np.uint8)
    
    def _add_ambient_lighting(self, image: np.ndarray, lighting_analysis: LightingAnalysis) -> np.ndarray:
        """Add ambient lighting effects to match the room."""
        result = image.astype(np.float32)
        
        # Create ambient light overlay based on room lighting
        h, w = image.shape[:2]
        light_x, light_y = lighting_analysis.dominant_light_direction
        
        # Create gradient overlay simulating ambient light
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        # Light intensity falls off with distance from light source
        center_x, center_y = w//2 + light_x * w//4, h//2 + light_y * h//4
        distance = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        max_distance = np.sqrt(w**2 + h**2)
        
        # Create subtle lighting gradient
        light_intensity = 1.0 + 0.15 * (1 - distance / max_distance) * lighting_analysis.ambient_intensity
        
        # Apply to each channel
        for c in range(3):
            result[:, :, c] *= light_intensity
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _create_blend_mask(self, foreground: np.ndarray, feather_radius: int = 5) -> np.ndarray:
        """Create a feathered blend mask for smooth edges."""
        # Create initial mask based on non-black pixels
        mask = np.where(np.sum(foreground, axis=2) > 30, 255, 0).astype(np.uint8)
        
        # Apply Gaussian blur for feathering
        if feather_radius > 0:
            mask = cv2.GaussianBlur(mask, (feather_radius*2+1, feather_radius*2+1), feather_radius/3)
        
        return mask
    
    def _poisson_blend(self, background: np.ndarray, foreground: np.ndarray,
                      position: Tuple[int, int], mask: np.ndarray) -> np.ndarray:
        """Perform Poisson blending for seamless integration."""
        try:
            # OpenCV's seamlessClone for Poisson blending
            x, y = position
            h, w = foreground.shape[:2]
            
            # Calculate center point for seamless clone
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Ensure center is within background bounds
            bg_h, bg_w = background.shape[:2]
            center_x = max(w//2, min(center_x, bg_w - w//2))
            center_y = max(h//2, min(center_y, bg_h - h//2))
            
            # Perform seamless cloning
            result = cv2.seamlessClone(
                foreground, background, mask, 
                (center_x, center_y), cv2.NORMAL_CLONE
            )
            
            return result
            
        except cv2.error as e:
            self.logger.warning(f"Poisson blending failed: {e}. Using alpha blending.")
            return self._alpha_blend(background, foreground, position, mask)
    
    def _alpha_blend(self, background: np.ndarray, foreground: np.ndarray,
                    position: Tuple[int, int], mask: np.ndarray) -> np.ndarray:
        """Fallback alpha blending method."""
        result = background.copy()
        x, y = position
        
        # Get dimensions
        fg_h, fg_w = foreground.shape[:2]
        bg_h, bg_w = background.shape[:2]
        
        # Calculate valid region
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(bg_w, x + fg_w)
        y2 = min(bg_h, y + fg_h)
        
        # Calculate corresponding foreground region
        fx1 = x1 - x
        fy1 = y1 - y
        fx2 = fx1 + (x2 - x1)
        fy2 = fy1 + (y2 - y1)
        
        if x2 > x1 and y2 > y1:
            # Extract regions
            bg_region = result[y1:y2, x1:x2]
            fg_region = foreground[fy1:fy2, fx1:fx2]
            mask_region = mask[fy1:fy2, fx1:fx2] / 255.0
            
            # Expand mask to 3 channels
            if len(mask_region.shape) == 2:
                mask_region = np.stack([mask_region] * 3, axis=2)
            
            # Alpha blend with proper type conversion
            mask_region = mask_region.astype(np.float32)
            fg_region = fg_region.astype(np.float32)
            bg_region = bg_region.astype(np.float32)
            
            blended = (mask_region * fg_region + (1 - mask_region) * bg_region)
            result[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
        
        return result
    
    def _add_realistic_shadows(self, image: np.ndarray, foreground: np.ndarray,
                              position: Tuple[int, int], lighting_analysis: LightingAnalysis,
                              wall_mask: np.ndarray) -> np.ndarray:
        """Add realistic shadows based on lighting analysis."""
        result = image.copy()
        
        # Calculate shadow offset based on light direction
        light_x, light_y = lighting_analysis.dominant_light_direction
        
        # Shadow offset (opposite to light direction)
        shadow_offset_x = int(-light_x * 20)  # Scale factor for shadow distance
        shadow_offset_y = int(-light_y * 20)
        
        # Create shadow mask from foreground
        shadow_mask = self._create_shadow_mask(foreground)
        
        # Position shadow
        shadow_x = position[0] + shadow_offset_x
        shadow_y = position[1] + shadow_offset_y
        
        # Apply shadow to image with stronger effect
        result = self._apply_shadow(result, shadow_mask, (shadow_x, shadow_y), wall_mask)
        
        # Add additional contact shadow directly under the TV for more realism
        contact_shadow = self._create_contact_shadow(foreground)
        contact_y = position[1] + foreground.shape[0] - 5  # Just below TV
        result = self._apply_shadow(result, contact_shadow, (position[0], contact_y), wall_mask, strength=0.5)
        
        return result
    
    def _create_shadow_mask(self, foreground: np.ndarray) -> np.ndarray:
        """Create a shadow mask from the foreground object."""
        # Create mask from non-transparent pixels
        if foreground.shape[2] == 4:
            alpha = foreground[:, :, 3]
            mask = (alpha > 50).astype(np.uint8) * 255
        else:
            # Use luminance to create mask
            gray = cv2.cvtColor(foreground, cv2.COLOR_RGB2GRAY)
            mask = (gray > 30).astype(np.uint8) * 255
        
        # Blur and distort for realistic shadow
        mask = cv2.GaussianBlur(mask, (15, 15), 5)
        
        # Apply perspective distortion to simulate shadow falling on wall
        h, w = mask.shape
        src_points = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        dst_points = np.array([[0, 0], [w, 0], [int(w*0.8), h], [int(w*0.2), h]], dtype=np.float32)
        
        transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        shadow_mask = cv2.warpPerspective(mask, transform_matrix, (w, h))
        
        return shadow_mask
    
    def _create_contact_shadow(self, foreground: np.ndarray) -> np.ndarray:
        """Create a contact shadow that appears directly under the object."""
        h, w = foreground.shape[:2]
        
        # Create a horizontal shadow strip
        shadow_height = max(10, h // 8)  # Shadow height based on object size
        contact_shadow = np.zeros((shadow_height, w), dtype=np.uint8)
        
        # Create gradient shadow (darker at top, fading down)
        for y in range(shadow_height):
            intensity = int(200 * (1 - y / shadow_height))  # Fade from 200 to 0
            contact_shadow[y, :] = intensity
        
        # Apply horizontal blur for softness
        contact_shadow = cv2.GaussianBlur(contact_shadow, (21, 7), 3)
        
        return contact_shadow
    
    def _apply_shadow(self, image: np.ndarray, shadow_mask: np.ndarray,
                     position: Tuple[int, int], wall_mask: np.ndarray, strength: float = 0.4) -> np.ndarray:
        """Apply shadow to the image."""
        result = image.copy()
        x, y = position
        
        # Get dimensions
        shadow_h, shadow_w = shadow_mask.shape
        img_h, img_w = image.shape[:2]
        
        # Calculate valid region
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_w, x + shadow_w)
        y2 = min(img_h, y + shadow_h)
        
        # Calculate corresponding shadow region
        sx1 = x1 - x
        sy1 = y1 - y
        sx2 = sx1 + (x2 - x1)
        sy2 = sy1 + (y2 - y1)
        
        if x2 > x1 and y2 > y1:
            # Extract regions
            img_region = result[y1:y2, x1:x2]
            shadow_region = shadow_mask[sy1:sy2, sx1:sx2]
            wall_region = wall_mask[y1:y2, x1:x2] if wall_mask is not None else np.ones_like(shadow_region)
            
            # Apply shadow only on wall areas
            shadow_factor = 1 - (shadow_region / 255.0 * strength)
            
            # Expand to 3 channels
            shadow_factor_3d = np.stack([shadow_factor] * 3, axis=2)
            wall_factor_3d = np.stack([wall_region / 255.0] * 3, axis=2)
            
            # Apply shadow with wall masking
            shadowed = img_region.astype(np.float32) * (shadow_factor_3d * wall_factor_3d + (1 - wall_factor_3d))
            result[y1:y2, x1:x2] = np.clip(shadowed, 0, 255).astype(np.uint8)
        
        return result
    
    def _apply_final_adjustments(self, image: np.ndarray, original_bg: np.ndarray,
                               position: Tuple[int, int], fg_dims: Tuple[int, int]) -> np.ndarray:
        """Apply final color and lighting adjustments."""
        result = image.copy()
        
        # Subtle color matching in the placement region
        x, y = position
        h, w = fg_dims
        
        # Get placement region
        img_h, img_w = image.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)
        
        if x2 > x1 and y2 > y1:
            # Extract regions
            placement_region = result[y1:y2, x1:x2]
            original_region = original_bg[y1:y2, x1:x2]
            
            # Subtle histogram matching for better integration
            matched_region = self._match_histograms(placement_region, original_region, strength=0.2)
            result[y1:y2, x1:x2] = matched_region
        
        return result
    
    def _match_histograms(self, source: np.ndarray, reference: np.ndarray, strength: float = 0.5) -> np.ndarray:
        """Apply subtle histogram matching between source and reference."""
        result = source.copy()
        
        for channel in range(3):
            # Calculate histograms
            src_hist, _ = np.histogram(source[:, :, channel], bins=256, range=(0, 256))
            ref_hist, _ = np.histogram(reference[:, :, channel], bins=256, range=(0, 256))
            
            # Calculate CDFs
            src_cdf = np.cumsum(src_hist) / np.sum(src_hist)
            ref_cdf = np.cumsum(ref_hist) / np.sum(ref_hist)
            
            # Create mapping
            mapping = np.zeros(256, dtype=np.uint8)
            for i in range(256):
                # Find closest CDF value in reference
                closest_idx = np.argmin(np.abs(ref_cdf - src_cdf[i]))
                mapping[i] = closest_idx
            
            # Apply mapping with strength factor
            mapped_channel = mapping[source[:, :, channel]]
            result[:, :, channel] = (
                strength * mapped_channel + (1 - strength) * source[:, :, channel]
            ).astype(np.uint8)
        
        return result
    
    def create_debug_visualization(self, original: np.ndarray, result: np.ndarray,
                                  lighting_analysis: LightingAnalysis) -> np.ndarray:
        """Create debug visualization showing lighting analysis and blending."""
        # Create a side-by-side comparison
        h, w = original.shape[:2]
        debug_img = np.zeros((h, w * 2, 3), dtype=np.uint8)
        
        # Left side: original with lighting overlay
        debug_img[:, :w] = original
        
        # Overlay lighting direction arrow
        center_x, center_y = w // 2, h // 2
        light_x, light_y = lighting_analysis.dominant_light_direction
        end_x = int(center_x + light_x * 100)
        end_y = int(center_y + light_y * 100)
        
        cv2.arrowedLine(debug_img, (center_x, center_y), (end_x, end_y), (0, 255, 0), 3)
        
        # Right side: result
        debug_img[:, w:] = result
        
        # Add text information
        cv2.putText(debug_img, "Original + Light Direction", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(debug_img, "Blended Result", (w + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(debug_img, f"Ambient: {lighting_analysis.ambient_intensity:.2f}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(debug_img, f"Color Temp: {lighting_analysis.color_temperature:.0f}K", 
                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return debug_img
