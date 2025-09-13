"""
Example Usage Script

This script demonstrates how to use the Wall AR Pipeline system
with sample images and different configuration options.
"""

import numpy as np
import cv2
from pathlib import Path
import logging
from PIL import Image, ImageDraw

from wall_ar_pipeline import WallARPipeline
from product_placement import ProductType
from utils import setup_logging, create_thumbnail
from config import config

def create_sample_room_image(width: int = 800, height: int = 600) -> np.ndarray:
    """Create a synthetic room image for testing."""
    # Create base room image
    room = np.ones((height, width, 3), dtype=np.uint8) * 200  # Light gray background
    
    # Add wall texture
    wall_region = room[50:height-100, 100:width-100]
    wall_region[:, :] = [220, 215, 210]  # Off-white wall
    
    # Add some furniture (dark rectangles at bottom)
    furniture_height = 80
    room[height-furniture_height:height, 50:200] = [101, 67, 33]  # Brown furniture
    room[height-furniture_height:height, width-250:width-50] = [101, 67, 33]
    
    # Add floor
    room[height-20:height, :] = [139, 119, 101]  # Wood floor
    
    # Add some lighting variation
    for i in range(height):
        brightness_factor = 0.8 + 0.4 * (i / height)  # Darker at top
        room[i, :] = np.clip(room[i, :] * brightness_factor, 0, 255)
    
    return room.astype(np.uint8)

def create_sample_tv_image(width: int = 200, height: int = 112) -> np.ndarray:
    """Create a synthetic TV image with transparency."""
    # Create RGBA image
    tv = np.zeros((height, width, 4), dtype=np.uint8)
    
    # TV bezel (dark gray)
    bezel_thickness = 8
    tv[bezel_thickness:-bezel_thickness, bezel_thickness:-bezel_thickness] = [30, 30, 30, 255]
    
    # Screen (black with slight blue tint)
    screen_margin = 12
    tv[screen_margin:-screen_margin, screen_margin:-screen_margin] = [10, 15, 25, 255]
    
    # Add some screen content (blue gradient)
    screen_region = tv[screen_margin:-screen_margin, screen_margin:-screen_margin]
    for i in range(screen_region.shape[0]):
        intensity = int(50 + 100 * (i / screen_region.shape[0]))
        screen_region[i, :] = [intensity//3, intensity//2, intensity, 255]
    
    return tv

def create_sample_painting_image(width: int = 150, height: int = 200) -> np.ndarray:
    """Create a synthetic painting image."""
    # Create RGB image
    painting = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Frame (golden brown)
    frame_thickness = 15
    painting[:frame_thickness, :] = [139, 117, 65]  # Top
    painting[-frame_thickness:, :] = [139, 117, 65]  # Bottom
    painting[:, :frame_thickness] = [139, 117, 65]  # Left
    painting[:, -frame_thickness:] = [139, 117, 65]  # Right
    
    # Canvas (off-white)
    canvas_region = painting[frame_thickness:-frame_thickness, frame_thickness:-frame_thickness]
    canvas_region[:, :] = [245, 240, 235]
    
    # Add simple artwork (abstract shapes)
    center_y, center_x = canvas_region.shape[0] // 2, canvas_region.shape[1] // 2
    
    # Blue circle
    cv2.circle(painting, (center_x + frame_thickness, center_y + frame_thickness - 20), 30, (100, 150, 200), -1)
    
    # Red rectangle
    cv2.rectangle(painting, 
                 (center_x - 25 + frame_thickness, center_y + 10 + frame_thickness),
                 (center_x + 25 + frame_thickness, center_y + 40 + frame_thickness),
                 (200, 100, 100), -1)
    
    return painting

def example_basic_usage():
    """Demonstrate basic usage of the pipeline."""
    print("=== Basic Usage Example ===")
    
    # Setup logging
    setup_logging("INFO")
    
    # Create sample images
    room_image = create_sample_room_image()
    tv_image = create_sample_tv_image()
    
    # Save sample images
    sample_dir = Path("sample_images")
    sample_dir.mkdir(exist_ok=True)
    
    cv2.imwrite(str(sample_dir / "room.jpg"), cv2.cvtColor(room_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(sample_dir / "tv.png"), cv2.cvtColor(tv_image, cv2.COLOR_RGBA2BGRA))
    
    # Initialize pipeline (without SAM for this example)
    pipeline = WallARPipeline(sam_checkpoint_path=None)
    
    # Process image
    try:
        result = pipeline.process_image(
            room_image_path=room_image,  # Pass numpy array directly
            product_image_path=tv_image,
            product_type=ProductType.TV,
            output_path=sample_dir / "result_basic.png",
            save_debug=True
        )
        
        print(f"✓ Processing successful!")
        print(f"  Confidence: {result.confidence_score:.2f}")
        print(f"  Processing time: {result.processing_time:.2f}s")
        print(f"  Output saved to: {sample_dir / 'result_basic.png'}")
        
    except Exception as e:
        print(f"✗ Processing failed: {e}")

def example_advanced_usage():
    """Demonstrate advanced features and configuration."""
    print("\n=== Advanced Usage Example ===")
    
    # Create sample images
    room_image = create_sample_room_image(1024, 768)
    painting_image = create_sample_painting_image()
    
    # Save sample images
    sample_dir = Path("sample_images")
    sample_dir.mkdir(exist_ok=True)
    
    cv2.imwrite(str(sample_dir / "room_large.jpg"), cv2.cvtColor(room_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(sample_dir / "painting.png"), cv2.cvtColor(painting_image, cv2.COLOR_RGB2BGR))
    
    # Configure advanced settings
    config["processing"].wall_confidence_threshold = 0.5  # Lower threshold
    config["placement"].default_painting_size_ratio = 0.08  # Smaller painting
    config["output"].save_intermediate_results = True
    
    # Initialize pipeline
    pipeline = WallARPipeline()
    
    try:
        result = pipeline.process_image(
            room_image_path=room_image,
            product_image_path=painting_image,
            product_type=ProductType.PAINTING,
            output_path=sample_dir / "result_advanced.png",
            placement_position=(0.3, 0.3),  # Custom position
            size_ratio=0.1,  # Custom size
            enable_shadows=True,
            enable_lighting_match=True,
            save_debug=True
        )
        
        print(f"✓ Advanced processing successful!")
        print(f"  Confidence: {result.confidence_score:.2f}")
        print(f"  Wall confidence: {result.segmentation_result.confidence:.2f}")
        print(f"  Placement confidence: {result.placement_result.confidence:.2f}")
        print(f"  Processing time: {result.processing_time:.2f}s")
        
        # Create comparison visualization
        comparison = pipeline.create_comparison_visualization(room_image, result)
        cv2.imwrite(str(sample_dir / "comparison.png"), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
        print(f"  Comparison saved to: {sample_dir / 'comparison.png'}")
        
    except Exception as e:
        print(f"✗ Advanced processing failed: {e}")

def example_batch_processing():
    """Demonstrate batch processing capabilities."""
    print("\n=== Batch Processing Example ===")
    
    # Create multiple sample configurations
    sample_dir = Path("sample_images")
    sample_dir.mkdir(exist_ok=True)
    
    # Create different room and product combinations
    configs = []
    
    for i in range(3):
        # Create varied room images
        room = create_sample_room_image(600 + i*100, 400 + i*50)
        room_path = sample_dir / f"room_{i}.jpg"
        cv2.imwrite(str(room_path), cv2.cvtColor(room, cv2.COLOR_RGB2BGR))
        
        # Alternate between TV and painting
        if i % 2 == 0:
            product = create_sample_tv_image(150 + i*20, 84 + i*11)
            product_path = sample_dir / f"tv_{i}.png"
            cv2.imwrite(str(product_path), cv2.cvtColor(product, cv2.COLOR_RGBA2BGRA))
            product_type = "tv"
        else:
            product = create_sample_painting_image(120 + i*15, 160 + i*20)
            product_path = sample_dir / f"painting_{i}.png"
            cv2.imwrite(str(product_path), cv2.cvtColor(product, cv2.COLOR_RGB2BGR))
            product_type = "painting"
        
        configs.append({
            "room_image_path": str(room_path),
            "product_image_path": str(product_path),
            "product_type": product_type,
            "enable_shadows": True,
            "enable_lighting_match": True
        })
    
    # Initialize pipeline
    pipeline = WallARPipeline()
    
    # Progress callback
    def progress_callback(current, total):
        print(f"  Processing {current + 1}/{total}...")
    
    try:
        # Process batch
        batch_results = pipeline.batch_process(
            input_configs=configs,
            output_directory=sample_dir / "batch_results",
            progress_callback=progress_callback
        )
        
        print(f"✓ Batch processing completed!")
        print(f"  Total items: {batch_results['total_items']}")
        print(f"  Successful: {batch_results['successful']}")
        print(f"  Failed: {batch_results['failed']}")
        print(f"  Success rate: {batch_results['success_rate']:.1%}")
        print(f"  Average confidence: {batch_results['average_confidence']:.2f}")
        print(f"  Average time per image: {batch_results['average_processing_time']:.2f}s")
        
        if batch_results['failed_items']:
            print("  Failed items:")
            for item in batch_results['failed_items']:
                print(f"    - Item {item['index']}: {item['error']}")
        
    except Exception as e:
        print(f"✗ Batch processing failed: {e}")

def example_quality_assessment():
    """Demonstrate quality assessment and validation."""
    print("\n=== Quality Assessment Example ===")
    
    from utils import calculate_image_quality_score, detect_room_type
    
    # Create sample room
    room_image = create_sample_room_image()
    
    # Assess input quality
    quality_score = calculate_image_quality_score(room_image)
    room_type = detect_room_type(room_image)
    
    print(f"Input image analysis:")
    print(f"  Quality score: {quality_score:.2f}")
    print(f"  Detected room type: {room_type}")
    
    # Process with quality monitoring
    pipeline = WallARPipeline()
    tv_image = create_sample_tv_image()
    
    try:
        result = pipeline.process_image(
            room_image_path=room_image,
            product_image_path=tv_image,
            product_type=ProductType.TV
        )
        
        print(f"\nProcessing results:")
        print(f"  Overall confidence: {result.confidence_score:.2f}")
        
        # Detailed confidence breakdown
        seg_conf = result.segmentation_result.confidence
        place_conf = result.placement_result.confidence
        
        print(f"  Segmentation confidence: {seg_conf:.2f}")
        print(f"  Placement confidence: {place_conf:.2f}")
        
        # Quality assessment
        if result.confidence_score > 0.8:
            print("  ✓ Excellent quality result")
        elif result.confidence_score > 0.6:
            print("  ⚠ Good quality result")
        else:
            print("  ⚠ Low quality result - consider adjusting parameters")
        
        # Get pipeline statistics
        stats = pipeline.get_statistics()
        print(f"\nPipeline statistics:")
        print(f"  Total processed: {stats['total_processed']}")
        print(f"  Successful placements: {stats['successful_placements']}")
        print(f"  Average confidence: {stats['average_confidence']:.2f}")
        
    except Exception as e:
        print(f"✗ Quality assessment failed: {e}")

def main():
    """Run all examples."""
    print("Wall AR Pipeline - Example Usage")
    print("=" * 50)
    
    try:
        # Run examples
        example_basic_usage()
        example_advanced_usage()
        example_batch_processing()
        example_quality_assessment()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("Check the 'sample_images' directory for results.")
        
    except KeyboardInterrupt:
        print("\nExamples interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")

if __name__ == "__main__":
    main()
