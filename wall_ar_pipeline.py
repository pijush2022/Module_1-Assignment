"""
Wall AR Pipeline - Main Orchestrator

This module provides the main pipeline that orchestrates wall segmentation,
product placement, and image blending for realistic AR try-on experiences.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Union, Dict, Any
from PIL import Image
import logging
from pathlib import Path
from dataclasses import dataclass
import json

from wall_segmentation import WallSegmenter, WallSegmentationResult
from product_placement import ProductPlacer, ProductType, PlacementResult
from image_blending import ImageBlender, BlendingResult
from config import config

@dataclass
class PipelineResult:
    """Complete pipeline result container."""
    final_image: np.ndarray
    segmentation_result: WallSegmentationResult
    placement_result: PlacementResult
    blending_result: BlendingResult
    processing_time: float
    confidence_score: float

class WallARPipelineError(Exception):
    """Custom exception for pipeline errors."""
    pass

class WallARPipeline:
    """
    Main pipeline for wall segmentation and product placement.
    
    This class orchestrates the entire process from room image input
    to final AR-enhanced output with realistic product placement.
    """
    
    def __init__(self, sam_checkpoint_path: Optional[str] = None):
        """
        Initialize the AR pipeline.
        
        Args:
            sam_checkpoint_path: Path to SAM model checkpoint (optional)
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.wall_segmenter = WallSegmenter(
            model_path=sam_checkpoint_path,  # U-Net model path (optional)
            device=config["model"].device
        )
        
        self.product_placer = ProductPlacer()
        self.image_blender = ImageBlender()
        
        # Processing statistics
        self.stats = {
            "total_processed": 0,
            "successful_placements": 0,
            "average_confidence": 0.0
        }
    
    def process_image(self,
                     room_image_path: Union[str, Path, np.ndarray],
                     product_image_path: Union[str, Path, np.ndarray],
                     product_type: Union[str, ProductType],
                     output_path: Optional[Union[str, Path]] = None,
                     placement_position: Optional[Tuple[float, float]] = None,
                     size_ratio: Optional[float] = None,
                     enable_shadows: bool = True,
                     enable_lighting_match: bool = True,
                     save_debug: bool = False) -> PipelineResult:
        """
        Process a room image and place a product on the wall.
        
        Args:
            room_image_path: Path to room image or numpy array
            product_image_path: Path to product image or numpy array
            product_type: Type of product ('tv', 'painting', 'frame')
            output_path: Path to save result (optional)
            placement_position: Relative position (0-1, 0-1) on wall
            size_ratio: Size as ratio of wall width
            enable_shadows: Whether to add realistic shadows
            enable_lighting_match: Whether to match lighting conditions
            save_debug: Whether to save debug visualizations
            
        Returns:
            PipelineResult with final image and processing metadata
        """
        import time
        start_time = time.time()
        
        try:
            # Load and validate images
            room_image = self._load_image(room_image_path)
            product_image = self._load_image(product_image_path)
            
            # Convert product type to enum
            if isinstance(product_type, str):
                product_type = ProductType(product_type.lower())
            
            # Resize room image if too large
            room_image = self._resize_if_needed(room_image)
            
            self.logger.info(f"Processing room image: {room_image.shape}")
            self.logger.info(f"Product type: {product_type.value}")
            
            # Step 1: Segment wall
            self.logger.info("Step 1: Segmenting wall...")
            segmentation_result = self.wall_segmenter.segment_wall(room_image)
            
            if segmentation_result.confidence < config["processing"].wall_confidence_threshold:
                self.logger.warning(f"Low wall segmentation confidence: {segmentation_result.confidence:.2f}")
            
            # Step 2: Place product
            self.logger.info("Step 2: Placing product...")
            placement_result = self.product_placer.place_product(
                room_image=room_image,
                product_image=product_image,
                wall_mask=segmentation_result.mask,
                wall_corners=segmentation_result.wall_corners,
                product_type=product_type,
                position=placement_position,
                size_ratio=size_ratio
            )
            
            # Step 3: Blend with lighting
            self.logger.info("Step 3: Blending with lighting adjustments...")
            blending_result = self.image_blender.blend_with_lighting(
                background=room_image,
                foreground=placement_result.placed_image[
                    placement_result.placement_box[1]:placement_result.placement_box[1] + placement_result.placement_box[3],
                    placement_result.placement_box[0]:placement_result.placement_box[0] + placement_result.placement_box[2]
                ],
                position=(placement_result.placement_box[0], placement_result.placement_box[1]),
                wall_mask=segmentation_result.mask,
                enable_shadows=enable_shadows,
                enable_lighting_match=enable_lighting_match
            )
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(
                segmentation_result, placement_result, blending_result
            )
            
            # Create final result
            processing_time = time.time() - start_time
            
            pipeline_result = PipelineResult(
                final_image=blending_result.blended_image,
                segmentation_result=segmentation_result,
                placement_result=placement_result,
                blending_result=blending_result,
                processing_time=processing_time,
                confidence_score=overall_confidence
            )
            
            # Save output if path provided
            if output_path:
                self._save_result(pipeline_result, output_path, save_debug)
            
            # Update statistics
            self._update_stats(pipeline_result)
            
            self.logger.info(f"Processing completed in {processing_time:.2f}s with confidence {overall_confidence:.2f}")
            
            return pipeline_result
            
        except Exception as e:
            self.logger.error(f"Pipeline processing failed: {str(e)}")
            raise WallARPipelineError(f"Processing failed: {str(e)}") from e
    
    def batch_process(self,
                     input_configs: list,
                     output_directory: Union[str, Path],
                     progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Process multiple images in batch.
        
        Args:
            input_configs: List of configuration dictionaries for each image
            output_directory: Directory to save results
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with batch processing results and statistics
        """
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        failed_items = []
        
        for i, config_item in enumerate(input_configs):
            try:
                if progress_callback:
                    progress_callback(i, len(input_configs))
                
                # Set output path
                output_name = f"result_{i:04d}.png"
                config_item["output_path"] = output_dir / output_name
                
                # Process image
                result = self.process_image(**config_item)
                results.append({
                    "index": i,
                    "config": config_item,
                    "result": result,
                    "success": True
                })
                
            except Exception as e:
                self.logger.error(f"Failed to process item {i}: {str(e)}")
                failed_items.append({
                    "index": i,
                    "config": config_item,
                    "error": str(e)
                })
        
        # Generate batch report
        batch_stats = {
            "total_items": len(input_configs),
            "successful": len(results),
            "failed": len(failed_items),
            "success_rate": len(results) / len(input_configs) if input_configs else 0,
            "average_confidence": np.mean([r["result"].confidence_score for r in results]) if results else 0,
            "average_processing_time": np.mean([r["result"].processing_time for r in results]) if results else 0,
            "failed_items": failed_items
        }
        
        # Save batch report
        report_path = output_dir / "batch_report.json"
        with open(report_path, 'w') as f:
            json.dump(batch_stats, f, indent=2, default=str)
        
        return batch_stats
    
    def _load_image(self, image_input: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Load image from path or return if already numpy array."""
        if isinstance(image_input, np.ndarray):
            return image_input
        
        image_path = Path(image_input)
        if not image_path.exists():
            raise WallARPipelineError(f"Image file not found: {image_path}")
        
        # Load with PIL for better format support
        pil_image = Image.open(image_path)
        
        # Convert to RGB if needed
        if pil_image.mode != 'RGB':
            if pil_image.mode == 'RGBA':
                # Keep RGBA for product images
                image_array = np.array(pil_image)
            else:
                pil_image = pil_image.convert('RGB')
                image_array = np.array(pil_image)
        else:
            image_array = np.array(pil_image)
        
        return image_array
    
    def _resize_if_needed(self, image: np.ndarray) -> np.ndarray:
        """Resize image if it exceeds maximum dimensions."""
        max_height, max_width = config["processing"].max_image_size
        height, width = image.shape[:2]
        
        if height > max_height or width > max_width:
            # Calculate scale factor
            scale = min(max_width / width, max_height / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            self.logger.info(f"Resizing image from {width}x{height} to {new_width}x{new_height}")
            
            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return image
    
    def _calculate_overall_confidence(self,
                                    segmentation_result: WallSegmentationResult,
                                    placement_result: PlacementResult,
                                    blending_result: BlendingResult) -> float:
        """Calculate overall confidence score for the pipeline result."""
        # Weight different components
        weights = {
            "segmentation": 0.4,
            "placement": 0.4,
            "blending": 0.2
        }
        
        # Segmentation confidence
        seg_confidence = segmentation_result.confidence
        
        # Placement confidence
        place_confidence = placement_result.confidence
        
        # Blending confidence (based on whether advanced features were used)
        blend_confidence = 0.8  # Base confidence
        if blending_result.lighting_adjusted:
            blend_confidence += 0.1
        if blending_result.shadow_added:
            blend_confidence += 0.1
        
        # Calculate weighted average
        overall_confidence = (
            weights["segmentation"] * seg_confidence +
            weights["placement"] * place_confidence +
            weights["blending"] * blend_confidence
        )
        
        return min(overall_confidence, 1.0)  # Cap at 1.0
    
    def _save_result(self, result: PipelineResult, output_path: Union[str, Path], save_debug: bool):
        """Save pipeline result to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save main result
        result_pil = Image.fromarray(result.final_image)
        result_pil.save(output_path, quality=config["output"].output_quality)
        
        if save_debug or config["output"].save_intermediate_results:
            # Save debug visualizations
            debug_dir = output_path.parent / f"{output_path.stem}_debug"
            debug_dir.mkdir(exist_ok=True)
            
            # Save segmentation visualization
            seg_vis = self.wall_segmenter.visualize_segmentation(
                result.segmentation_result.mask, result.segmentation_result
            )
            Image.fromarray(seg_vis).save(debug_dir / "segmentation.png")
            
            # Save placement visualization
            placement_vis = result.placement_result.placed_image
            Image.fromarray(placement_vis).save(debug_dir / "placement.png")
            
            # Save metadata
            metadata = {
                "processing_time": result.processing_time,
                "confidence_score": result.confidence_score,
                "segmentation_confidence": result.segmentation_result.confidence,
                "placement_confidence": result.placement_result.confidence,
                "wall_corners": result.segmentation_result.wall_corners,
                "placement_box": result.placement_result.placement_box
            }
            
            with open(debug_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
    
    def _update_stats(self, result: PipelineResult):
        """Update processing statistics."""
        self.stats["total_processed"] += 1
        
        if result.confidence_score > 0.5:  # Consider successful if confidence > 0.5
            self.stats["successful_placements"] += 1
        
        # Update running average confidence
        total = self.stats["total_processed"]
        current_avg = self.stats["average_confidence"]
        new_confidence = result.confidence_score
        
        self.stats["average_confidence"] = (current_avg * (total - 1) + new_confidence) / total
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.stats.copy()
    
    def create_comparison_visualization(self, 
                                     original_image: np.ndarray,
                                     result: PipelineResult) -> np.ndarray:
        """Create a before/after comparison visualization."""
        # Resize images to same height if needed
        orig_h, orig_w = original_image.shape[:2]
        result_h, result_w = result.final_image.shape[:2]
        
        if orig_h != result_h:
            scale = result_h / orig_h
            new_w = int(orig_w * scale)
            original_resized = cv2.resize(original_image, (new_w, result_h))
        else:
            original_resized = original_image
        
        # Create side-by-side comparison
        comparison = np.hstack([original_resized, result.final_image])
        
        # Add text labels
        cv2.putText(comparison, "Original", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "AR Enhanced", (original_resized.shape[1] + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add confidence score
        cv2.putText(comparison, f"Confidence: {result.confidence_score:.2f}", 
                   (original_resized.shape[1] + 10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return comparison

def main():
    """Example usage of the Wall AR Pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Wall AR Pipeline")
    parser.add_argument("--room", required=True, help="Path to room image")
    parser.add_argument("--product", required=True, help="Path to product image")
    parser.add_argument("--type", required=True, choices=["tv", "painting", "frame"], 
                       help="Product type")
    parser.add_argument("--output", required=True, help="Output path")
    parser.add_argument("--sam-checkpoint", help="Path to SAM checkpoint")
    parser.add_argument("--debug", action="store_true", help="Save debug visualizations")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create pipeline
    pipeline = WallARPipeline(sam_checkpoint_path=args.sam_checkpoint)
    
    # Process image
    result = pipeline.process_image(
        room_image_path=args.room,
        product_image_path=args.product,
        product_type=args.type,
        output_path=args.output,
        save_debug=args.debug
    )
    
    print(f"Processing completed successfully!")
    print(f"Confidence score: {result.confidence_score:.2f}")
    print(f"Processing time: {result.processing_time:.2f}s")

if __name__ == "__main__":
    main()
