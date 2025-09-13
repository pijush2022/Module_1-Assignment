"""
Product Placement Module

This module handles the placement of products (TVs, paintings, frames) onto detected wall surfaces
with proper scaling, perspective correction, and realistic positioning.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Union
from PIL import Image
import logging
from dataclasses import dataclass
from enum import Enum

class ProductType(Enum):
    """Enumeration for different product types."""
    TV = "tv"
    PAINTING = "painting"
    FRAME = "frame"

@dataclass
class PlacementResult:
    """Result container for product placement."""
    placed_image: np.ndarray
    placement_box: Tuple[int, int, int, int]  # x, y, width, height
    transform_matrix: np.ndarray
    confidence: float

class ProductPlacementError(Exception):
    """Custom exception for product placement errors."""
    pass

class ProductPlacer:
    """
    Main class for placing products on walls with realistic perspective and scaling.
    
    Features:
    - Automatic perspective correction based on wall geometry
    - Intelligent scaling based on product type and wall size
    - Collision detection to avoid furniture overlap
    - Lighting-aware placement for realistic integration
    """
    
    def __init__(self):
        """Initialize the product placer."""
        self.logger = logging.getLogger(__name__)
    
    def place_product(self, 
                     room_image: np.ndarray,
                     product_image: np.ndarray,
                     wall_mask: np.ndarray,
                     wall_corners: List[Tuple[int, int]],
                     product_type: ProductType,
                     position: Optional[Tuple[float, float]] = None,
                     size_ratio: Optional[float] = None) -> PlacementResult:
        """
        Place a product on the wall with proper perspective and scaling.
        
        Args:
            room_image: Original room image
            product_image: Product image with transparency (RGBA)
            wall_mask: Binary mask of the wall region
            wall_corners: Corner points of the wall
            product_type: Type of product being placed
            position: Relative position on wall (0-1, 0-1), None for auto
            size_ratio: Size as ratio of wall width, None for default
            
        Returns:
            PlacementResult with the final composite image and metadata
        """
        # Validate inputs
        self._validate_inputs(room_image, product_image, wall_mask, wall_corners)
        
        # Determine optimal placement position
        if position is None:
            position = self._find_optimal_position(room_image, wall_mask, product_type)
        
        # Determine appropriate size
        if size_ratio is None:
            size_ratio = self._get_default_size_ratio(product_type)
        
        # Calculate wall plane geometry
        wall_plane = self._estimate_wall_plane(wall_corners, room_image.shape[:2])
        
        # Calculate product dimensions and position
        product_dims = self._calculate_product_dimensions(
            wall_mask, wall_corners, size_ratio, product_type
        )
        
        # Apply perspective correction to product
        corrected_product = self._apply_perspective_correction(
            product_image, wall_plane, product_dims
        )
        
        # Calculate final placement position
        placement_pos = self._calculate_placement_position(
            wall_mask, wall_corners, position, corrected_product.shape[:2]
        )
        
        # Check for collisions with furniture/objects
        collision_check = self._check_collisions(
            room_image, wall_mask, placement_pos, corrected_product.shape[:2]
        )
        
        if not collision_check:
            # Adjust position to avoid collisions
            placement_pos = self._adjust_for_collisions(
                room_image, wall_mask, placement_pos, corrected_product.shape[:2]
            )
        
        # Composite the product onto the room image
        final_image = self._composite_product(
            room_image, corrected_product, placement_pos
        )
        
        # Calculate transform matrix for reference
        transform_matrix = self._calculate_transform_matrix(
            product_image.shape[:2], corrected_product.shape[:2], placement_pos
        )
        
        # Calculate confidence score
        confidence = self._calculate_placement_confidence(
            wall_mask, placement_pos, corrected_product.shape[:2], collision_check
        )
        
        return PlacementResult(
            placed_image=final_image,
            placement_box=(placement_pos[0], placement_pos[1], 
                          corrected_product.shape[1], corrected_product.shape[0]),
            transform_matrix=transform_matrix,
            confidence=confidence
        )
    
    def _validate_inputs(self, room_image: np.ndarray, product_image: np.ndarray,
                        wall_mask: np.ndarray, wall_corners: List[Tuple[int, int]]):
        """Validate input parameters."""
        if room_image.shape[:2] != wall_mask.shape:
            raise ProductPlacementError("Room image and wall mask dimensions don't match")
        
        if len(wall_corners) < 3:
            raise ProductPlacementError("Need at least 3 wall corners for placement")
        
        if product_image.shape[2] not in [3, 4]:
            raise ProductPlacementError("Product image must be RGB or RGBA")
    
    def _find_optimal_position(self, room_image: np.ndarray, wall_mask: np.ndarray,
                              product_type: ProductType) -> Tuple[float, float]:
        """Find optimal position on wall for product placement with furniture avoidance."""
        # Analyze wall regions to find best placement area
        wall_regions = self._analyze_wall_regions(wall_mask)
        
        # Detect furniture and obstacles to avoid overlap
        furniture_regions = self._detect_furniture_regions(room_image, wall_mask)
        
        # Find clear wall areas
        clear_positions = self._find_clear_wall_positions(wall_mask, furniture_regions)
        
        # Different strategies based on product type
        if product_type == ProductType.TV:
            # TVs typically go in center, at eye level, avoiding furniture
            optimal_pos = self._find_tv_position(clear_positions, wall_mask)
            return optimal_pos if optimal_pos else (0.5, 0.4)
        elif product_type == ProductType.PAINTING:
            # Paintings can be more flexible in positioning
            optimal_pos = self._find_artwork_position(clear_positions, wall_mask)
            return optimal_pos if optimal_pos else (0.5, 0.35)
        else:  # FRAME
            # Frames similar to paintings but can be smaller
            optimal_pos = self._find_artwork_position(clear_positions, wall_mask)
            return optimal_pos if optimal_pos else (0.5, 0.4)
    
    def _get_default_size_ratio(self, product_type: ProductType) -> float:
        """Get default size ratio based on product type with realistic scaling."""
        size_ratios = {
            ProductType.TV: 0.18,      # Larger TVs for modern rooms
            ProductType.PAINTING: 0.14, # Medium-sized artwork
            ProductType.FRAME: 0.12     # Smaller frames
        }
        return size_ratios.get(product_type, 0.12)
    
    def _detect_furniture_regions(self, room_image: np.ndarray, wall_mask: np.ndarray) -> np.ndarray:
        """Detect furniture and obstacles in the room to avoid overlap."""
        # Convert to different color spaces for furniture detection
        gray = cv2.cvtColor(room_image, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(room_image, cv2.COLOR_RGB2HSV)
        
        # Edge detection to find furniture boundaries
        edges = cv2.Canny(gray, 50, 150)
        
        # Morphological operations to connect furniture edges
        kernel = np.ones((5, 5), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours (potential furniture)
        contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create furniture mask
        furniture_mask = np.zeros_like(gray)
        
        # Filter contours by size and position (furniture is typically large and not on walls)
        h, w = room_image.shape[:2]
        min_furniture_area = h * w * 0.01  # At least 1% of image
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_furniture_area:
                # Check if contour is mostly not on wall (furniture detection)
                contour_mask = np.zeros_like(gray)
                cv2.fillPoly(contour_mask, [contour], 255)
                
                # If less than 30% overlaps with wall, it's likely furniture
                wall_overlap = np.sum((contour_mask > 0) & (wall_mask > 0))
                total_contour = np.sum(contour_mask > 0)
                
                if total_contour > 0 and (wall_overlap / total_contour) < 0.3:
                    cv2.fillPoly(furniture_mask, [contour], 255)
        
        return furniture_mask
    
    def _find_clear_wall_positions(self, wall_mask: np.ndarray, furniture_mask: np.ndarray) -> List[Tuple[float, float]]:
        """Find clear positions on wall that don't overlap with furniture."""
        # Create combined obstacle mask (inverted wall + furniture)
        obstacles = (~(wall_mask > 0).astype(bool) | (furniture_mask > 0)).astype(np.uint8) * 255
        
        # Find clear areas using distance transform
        clear_areas = cv2.distanceTransform(255 - obstacles, cv2.DIST_L2, 5)
        
        # Find local maxima (best clear positions)
        h, w = clear_areas.shape
        clear_positions = []
        
        # Sample grid positions and evaluate clearance
        for y in range(h // 10, h, h // 5):  # Sample every 20% of height
            for x in range(w // 10, w, w // 5):  # Sample every 20% of width
                if clear_areas[y, x] > 20:  # Minimum clearance threshold
                    # Convert to relative coordinates
                    rel_x = x / w
                    rel_y = y / h
                    clear_positions.append((rel_x, rel_y))
        
        return clear_positions
    
    def _find_tv_position(self, clear_positions: List[Tuple[float, float]], wall_mask: np.ndarray) -> Optional[Tuple[float, float]]:
        """Find optimal TV position - center, eye level, with good clearance."""
        if not clear_positions:
            return None
        
        # Prefer positions near center and at eye level (40% height)
        ideal_x, ideal_y = 0.5, 0.4
        
        best_pos = None
        best_score = -1
        
        for pos_x, pos_y in clear_positions:
            # Score based on distance from ideal position
            distance = np.sqrt((pos_x - ideal_x)**2 + (pos_y - ideal_y)**2)
            score = 1.0 - min(distance, 1.0)  # Closer to ideal = higher score
            
            # Bonus for being in center horizontally
            if 0.3 <= pos_x <= 0.7:
                score += 0.2
            
            # Bonus for being at good height
            if 0.3 <= pos_y <= 0.5:
                score += 0.2
            
            if score > best_score:
                best_score = score
                best_pos = (pos_x, pos_y)
        
        return best_pos
    
    def _find_artwork_position(self, clear_positions: List[Tuple[float, float]], wall_mask: np.ndarray) -> Optional[Tuple[float, float]]:
        """Find optimal artwork position - flexible but avoiding furniture."""
        if not clear_positions:
            return None
        
        # Artwork can be more flexible in positioning
        best_pos = None
        best_score = -1
        
        for pos_x, pos_y in clear_positions:
            score = 1.0
            
            # Prefer upper portions of wall for artwork
            if pos_y < 0.6:
                score += 0.3
            
            # Prefer not too close to edges
            if 0.2 <= pos_x <= 0.8:
                score += 0.2
            
            # Avoid very top or bottom
            if 0.1 <= pos_y <= 0.8:
                score += 0.1
            
            if score > best_score:
                best_score = score
                best_pos = (pos_x, pos_y)
        
        return best_pos
    
    def _estimate_wall_plane(self, wall_corners: List[Tuple[int, int]], 
                           image_shape: Tuple[int, int]) -> np.ndarray:
        """Estimate the 3D plane of the wall from corner points."""
        # Convert corners to numpy array
        corners = np.array(wall_corners, dtype=np.float32)
        
        # If we have exactly 4 corners, use them directly
        if len(corners) == 4:
            # Sort corners to get proper rectangle
            corners = self._sort_rectangle_corners(corners)
        else:
            # Fit a rectangle to the corner points
            rect = cv2.minAreaRect(corners)
            corners = cv2.boxPoints(rect).astype(np.float32)
        
        # Estimate perspective transformation
        # Assume wall is rectangular in 3D space
        h, w = image_shape
        
        # Define ideal rectangle (frontal view)
        ideal_corners = np.array([
            [0, 0],
            [w//3, 0],
            [w//3, h//3],
            [0, h//3]
        ], dtype=np.float32)
        
        # Calculate perspective transform matrix
        perspective_matrix = cv2.getPerspectiveTransform(corners, ideal_corners)
        
        return perspective_matrix
    
    def _sort_rectangle_corners(self, corners: np.ndarray) -> np.ndarray:
        """Sort rectangle corners in consistent order: top-left, top-right, bottom-right, bottom-left."""
        # Calculate center point
        center = np.mean(corners, axis=0)
        
        # Sort by angle from center
        angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        
        return corners[sorted_indices]
    
    def _calculate_product_dimensions(self, wall_mask: np.ndarray, 
                                    wall_corners: List[Tuple[int, int]],
                                    size_ratio: float, 
                                    product_type: ProductType) -> Tuple[int, int]:
        """Calculate appropriate product dimensions based on wall size."""
        # Estimate wall width from corners
        corners = np.array(wall_corners)
        
        if len(corners) >= 2:
            # Calculate distances between consecutive corners
            distances = []
            for i in range(len(corners)):
                p1 = corners[i]
                p2 = corners[(i + 1) % len(corners)]
                dist = np.linalg.norm(p2 - p1)
                distances.append(dist)
            
            # Use maximum distance as wall width estimate
            wall_width = max(distances)
        else:
            # Fallback: use mask bounding box
            wall_coords = np.where(wall_mask > 0)
            if len(wall_coords[0]) > 0:
                wall_width = np.max(wall_coords[1]) - np.min(wall_coords[1])
            else:
                wall_width = 300  # Default fallback
        
        # Calculate product width based on size ratio
        product_width = int(wall_width * size_ratio)
        
        # Calculate height based on typical aspect ratios
        aspect_ratios = {
            ProductType.TV: 16/9,  # Modern TV aspect ratio
            ProductType.PAINTING: 4/3,  # Common painting ratio
            ProductType.FRAME: 3/4  # Portrait frame
        }
        
        aspect_ratio = aspect_ratios.get(product_type, 4/3)
        product_height = int(product_width / aspect_ratio)
        
        return (product_width, product_height)
    
    def _apply_perspective_correction(self, product_image: np.ndarray,
                                    wall_plane: np.ndarray,
                                    target_dims: Tuple[int, int]) -> np.ndarray:
        """Apply perspective correction to product image to match wall plane."""
        # Resize product to target dimensions first
        resized_product = cv2.resize(product_image, target_dims)
        
        # For frames and paintings, apply minimal perspective correction
        # to avoid severe distortion while maintaining some realism
        h, w = resized_product.shape[:2]
        
        # Apply subtle perspective effect (much less aggressive)
        try:
            # Create subtle perspective transformation
            # Only apply minor keystone correction, not full perspective warp
            
            # Calculate a subtle perspective factor based on image position
            perspective_factor = 0.05  # Very subtle effect (was causing severe distortion)
            
            src_corners = np.array([
                [0, 0],
                [w, 0], 
                [w, h],
                [0, h]
            ], dtype=np.float32)
            
            # Apply minimal perspective distortion
            dst_corners = np.array([
                [w * perspective_factor, h * perspective_factor],
                [w * (1 - perspective_factor), 0],
                [w * (1 - perspective_factor), h * (1 - perspective_factor)],
                [w * perspective_factor, h]
            ], dtype=np.float32)
            
            # Create and apply subtle perspective transform
            perspective_transform = cv2.getPerspectiveTransform(src_corners, dst_corners)
            
            # Apply transformation with same output size to avoid distortion
            corrected_product = cv2.warpPerspective(
                resized_product, 
                perspective_transform,
                (w, h)  # Keep same dimensions
            )
            
            return corrected_product
            
        except cv2.error:
            # Fallback: return resized product without perspective correction
            self.logger.warning("Perspective correction failed, using original product")
            return resized_product
    
    def _calculate_placement_position(self, wall_mask: np.ndarray,
                                    wall_corners: List[Tuple[int, int]],
                                    relative_position: Tuple[float, float],
                                    product_dims: Tuple[int, int]) -> Tuple[int, int]:
        """Calculate absolute pixel position for product placement."""
        # Get wall bounding box
        wall_coords = np.where(wall_mask > 0)
        
        if len(wall_coords[0]) == 0:
            raise ProductPlacementError("Empty wall mask")
        
        wall_y_min, wall_y_max = np.min(wall_coords[0]), np.max(wall_coords[0])
        wall_x_min, wall_x_max = np.min(wall_coords[1]), np.max(wall_coords[1])
        
        wall_width = wall_x_max - wall_x_min
        wall_height = wall_y_max - wall_y_min
        
        # Calculate absolute position
        rel_x, rel_y = relative_position
        product_h, product_w = product_dims
        
        # Center the product at the relative position
        abs_x = wall_x_min + int(rel_x * wall_width - product_w // 2)
        abs_y = wall_y_min + int(rel_y * wall_height - product_h // 2)
        
        # Ensure product stays within wall bounds
        abs_x = max(wall_x_min, min(abs_x, wall_x_max - product_w))
        abs_y = max(wall_y_min, min(abs_y, wall_y_max - product_h))
        
        return (abs_x, abs_y)
    
    def _check_collisions(self, room_image: np.ndarray, wall_mask: np.ndarray,
                         placement_pos: Tuple[int, int], 
                         product_dims: Tuple[int, int]) -> bool:
        """Check if product placement would collide with furniture or other objects."""
        x, y = placement_pos
        h, w = product_dims
        
        # Create product placement region
        product_region = np.zeros_like(wall_mask)
        
        # Ensure coordinates are within image bounds
        img_h, img_w = room_image.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)
        
        if x2 <= x1 or y2 <= y1:
            return False  # Invalid placement
        
        product_region[y1:y2, x1:x2] = 1
        
        # Check if product region overlaps with wall
        wall_overlap = np.sum(product_region & wall_mask) / np.sum(product_region)
        
        # Product should be mostly on the wall (>80% overlap)
        return wall_overlap > 0.8
    
    def _adjust_for_collisions(self, room_image: np.ndarray, wall_mask: np.ndarray,
                              original_pos: Tuple[int, int],
                              product_dims: Tuple[int, int]) -> Tuple[int, int]:
        """Adjust placement position to avoid collisions."""
        # Try different positions around the original position
        x, y = original_pos
        h, w = product_dims
        
        # Define search offsets (relative to product size)
        offsets = [
            (0, 0),  # Original position
            (w//4, 0), (-w//4, 0),  # Horizontal shifts
            (0, h//4), (0, -h//4),  # Vertical shifts
            (w//4, h//4), (-w//4, -h//4),  # Diagonal shifts
            (w//4, -h//4), (-w//4, h//4)
        ]
        
        for dx, dy in offsets:
            test_pos = (x + dx, y + dy)
            if self._check_collisions(room_image, wall_mask, test_pos, product_dims):
                return test_pos
        
        # If no good position found, return original
        return original_pos
    
    def _composite_product(self, room_image: np.ndarray, product_image: np.ndarray,
                          position: Tuple[int, int]) -> np.ndarray:
        """Composite product image onto room image with proper blending."""
        result = room_image.copy()
        x, y = position
        
        # Handle RGBA vs RGB product images
        if product_image.shape[2] == 4:
            # Product has alpha channel
            product_rgb = product_image[:, :, :3]
            alpha = product_image[:, :, 3] / 255.0
        else:
            # No alpha channel, create one based on black pixels
            product_rgb = product_image
            # Assume black pixels are transparent
            alpha = np.where(
                np.sum(product_rgb, axis=2) > 30,  # Not pure black
                1.0, 0.0
            )
        
        # Get dimensions
        prod_h, prod_w = product_rgb.shape[:2]
        room_h, room_w = room_image.shape[:2]
        
        # Calculate valid region for compositing
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(room_w, x + prod_w)
        y2 = min(room_h, y + prod_h)
        
        # Calculate corresponding product region
        px1 = x1 - x
        py1 = y1 - y
        px2 = px1 + (x2 - x1)
        py2 = py1 + (y2 - y1)
        
        if x2 > x1 and y2 > y1 and px2 > px1 and py2 > py1:
            # Extract regions
            room_region = result[y1:y2, x1:x2]
            product_region = product_rgb[py1:py2, px1:px2]
            alpha_region = alpha[py1:py2, px1:px2]
            
            # Expand alpha to 3 channels
            if len(alpha_region.shape) == 2:
                alpha_region = np.stack([alpha_region] * 3, axis=2)
            
            # Blend using alpha compositing
            blended = (alpha_region * product_region + 
                      (1 - alpha_region) * room_region).astype(np.uint8)
            
            # Place back into result
            result[y1:y2, x1:x2] = blended
        
        return result
    
    def _calculate_transform_matrix(self, original_dims: Tuple[int, int],
                                  final_dims: Tuple[int, int],
                                  position: Tuple[int, int]) -> np.ndarray:
        """Calculate transformation matrix from original to final placement."""
        orig_h, orig_w = original_dims
        final_h, final_w = final_dims
        x, y = position
        
        # Scale factors
        scale_x = final_w / orig_w
        scale_y = final_h / orig_h
        
        # Create transformation matrix
        transform = np.array([
            [scale_x, 0, x],
            [0, scale_y, y],
            [0, 0, 1]
        ], dtype=np.float32)
        
        return transform
    
    def _calculate_placement_confidence(self, wall_mask: np.ndarray,
                                      position: Tuple[int, int],
                                      product_dims: Tuple[int, int],
                                      collision_free: bool) -> float:
        """Calculate confidence score for the placement."""
        confidence_factors = []
        
        # Factor 1: Wall coverage (how much of product is on wall)
        wall_coverage = self._calculate_wall_coverage(wall_mask, position, product_dims)
        confidence_factors.append(wall_coverage)
        
        # Factor 2: Collision avoidance
        collision_factor = 1.0 if collision_free else 0.5
        confidence_factors.append(collision_factor)
        
        # Factor 3: Position reasonableness (not too close to edges)
        position_factor = self._calculate_position_reasonableness(
            wall_mask, position, product_dims
        )
        confidence_factors.append(position_factor)
        
        return np.mean(confidence_factors)
    
    def _calculate_wall_coverage(self, wall_mask: np.ndarray,
                               position: Tuple[int, int],
                               product_dims: Tuple[int, int]) -> float:
        """Calculate what fraction of the product is on the wall."""
        x, y = position
        h, w = product_dims
        
        # Create product region
        img_h, img_w = wall_mask.shape
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        product_area = (x2 - x1) * (y2 - y1)
        wall_overlap_area = np.sum(wall_mask[y1:y2, x1:x2])
        
        return wall_overlap_area / product_area if product_area > 0 else 0.0
    
    def _calculate_position_reasonableness(self, wall_mask: np.ndarray,
                                         position: Tuple[int, int],
                                         product_dims: Tuple[int, int]) -> float:
        """Calculate how reasonable the position is (not too close to edges)."""
        # Get wall bounds
        wall_coords = np.where(wall_mask > 0)
        if len(wall_coords[0]) == 0:
            return 0.0
        
        wall_y_min, wall_y_max = np.min(wall_coords[0]), np.max(wall_coords[0])
        wall_x_min, wall_x_max = np.min(wall_coords[1]), np.max(wall_coords[1])
        
        x, y = position
        h, w = product_dims
        
        # Calculate distances to wall edges
        dist_left = x - wall_x_min
        dist_right = wall_x_max - (x + w)
        dist_top = y - wall_y_min
        dist_bottom = wall_y_max - (y + h)
        
        # Minimum reasonable distance (10% of wall dimension)
        wall_width = wall_x_max - wall_x_min
        wall_height = wall_y_max - wall_y_min
        min_margin_x = wall_width * 0.1
        min_margin_y = wall_height * 0.1
        
        # Calculate reasonableness factors
        factors = []
        factors.append(1.0 if dist_left >= min_margin_x else dist_left / min_margin_x)
        factors.append(1.0 if dist_right >= min_margin_x else dist_right / min_margin_x)
        factors.append(1.0 if dist_top >= min_margin_y else dist_top / min_margin_y)
        factors.append(1.0 if dist_bottom >= min_margin_y else dist_bottom / min_margin_y)
        
        return np.mean(factors)
    
    def _analyze_wall_regions(self, wall_mask: np.ndarray) -> dict:
        """Analyze wall regions to identify optimal placement areas."""
        # This is a placeholder for more sophisticated analysis
        # Could include furniture detection, lighting analysis, etc.
        return {
            "optimal_regions": [],
            "avoid_regions": [],
            "lighting_zones": []
        }
