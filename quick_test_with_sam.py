"""
Quick test script that uses SAM ViT-H model if available, falls back to traditional methods otherwise.
"""

from pathlib import Path
import sys

def test_with_sam():
    """Test the pipeline with SAM enabled if available."""
    
    # Check organized structure
    walls_dir = Path("input/walls/")
    tvs_dir = Path("input/products/tvs/")
    
    if not walls_dir.exists() or not tvs_dir.exists():
        print("❌ Organized folder structure not found.")
        return False
    
    # Find images
    wall_images = list(walls_dir.glob("*.jpg"))
    tv_images = list(tvs_dir.glob("*.png"))
    
    if not wall_images or not tv_images:
        print("❌ No sample images found.")
        return False
    
    print(f"🖼️ Using: {wall_images[1].name} + {tv_images[0].name}")
    
    # Check if SAM model is available
    sam_path = Path("models/sam_vit_h_4b8939.pth")
    if sam_path.exists():
        print("🤖 SAM model found - using enhanced segmentation")
        checkpoint_path = str(sam_path)
    else:
        print("🔧 SAM model not found - using traditional CV methods")
        checkpoint_path = None
    
    try:
        from wall_ar_pipeline import WallARPipeline
        from product_placement import ProductType
        
        # Initialize pipeline with SAM if available
        pipeline = WallARPipeline(sam_checkpoint_path=checkpoint_path)
        
        result = pipeline.process_image(
            room_image_path=str(wall_images[0]),
            product_image_path=str(tv_images[0]),
            product_type=ProductType.TV,
            output_path="output/results/sam_test_result.png",
            enable_shadows=True,
            enable_lighting_match=True,
            save_debug=True
        )
        
        print(f"✅ Success!")
        print(f"   Segmentation confidence: {result.segmentation_result.confidence:.2f}")
        print(f"   Placement confidence: {result.placement_result.confidence:.2f}")
        print(f"   Overall confidence: {result.confidence_score:.2f}")
        print(f"   Processing time: {result.processing_time:.2f}s")
        print(f"   Output: output/results/sam_test_result.png")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_with_sam()
    sys.exit(0 if success else 1)
