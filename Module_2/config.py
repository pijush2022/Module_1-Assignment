"""
Configuration file for Product Placement Pipeline
Contains model settings, paths, and optimization parameters
"""

import torch
from pathlib import Path

class Config:
    """Configuration class for the product placement pipeline"""
    
    # Device configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
    
    # Model configurations
    BASE_MODEL = "runwayml/stable-diffusion-v1-5"
    CONTROLNET_DEPTH_MODEL = "lllyasviel/sd-controlnet-depth"
    CONTROLNET_CANNY_MODEL = "lllyasviel/sd-controlnet-canny"
    CONTROLNET_OPENPOSE_MODEL = "lllyasviel/sd-controlnet-openpose"
    
    # Generation parameters
    DEFAULT_STEPS = 20
    DEFAULT_GUIDANCE_SCALE = 7.5
    DEFAULT_CONTROLNET_SCALE = 1.0
    IMAGE_RESOLUTION = 512
    
    # Product size mappings (in pixels for reference)
    PRODUCT_SIZES = {
        "tv": {
            "32 inch": {"width": 120, "height": 68},
            "42 inch": {"width": 158, "height": 89},
            "55 inch": {"width": 207, "height": 116},
            "65 inch": {"width": 244, "height": 137},
            "75 inch": {"width": 282, "height": 159}
        },
        "painting": {
            "small": {"width": 80, "height": 60},
            "medium": {"width": 120, "height": 90},
            "large": {"width": 160, "height": 120},
            "extra large": {"width": 200, "height": 150}
        }
    }
    
    # Legacy aliases for backward compatibility
    TV_SIZES = PRODUCT_SIZES["tv"]
    PAINTING_SIZES = PRODUCT_SIZES["painting"]
    
    # Prompt templates
    PROMPTS = {
        "tv": {
            "base": "a modern {size} flat screen TV mounted on the wall",
            "enhanced": "a sleek {size} {brand} TV mounted on the wall, realistic lighting, high quality, detailed, professional photography",
            "negative": "blurry, low quality, distorted, unrealistic, floating, disconnected from wall, bad proportions"
        },
        "painting": {
            "base": "a beautiful {size} framed painting hanging on the wall",
            "enhanced": "an elegant {size} {style} painting in ornate frame hanging on the wall, museum quality, perfect lighting, high resolution",
            "negative": "blurry, low quality, crooked, floating, bad frame, unrealistic shadows"
        },
        "mirror": {
            "base": "a {size} mirror mounted on the wall",
            "enhanced": "a stylish {size} {style} mirror with elegant frame mounted on the wall, perfect reflection, high quality",
            "negative": "distorted reflection, cracked, floating, unrealistic, bad proportions"
        }
    }
    
    # Wall detection regions (as percentage of image dimensions)
    WALL_REGIONS = {
        "center": {"x": 0.25, "y": 0.2, "width": 0.5, "height": 0.4},
        "left": {"x": 0.1, "y": 0.2, "width": 0.4, "height": 0.4},
        "right": {"x": 0.5, "y": 0.2, "width": 0.4, "height": 0.4},
        "upper": {"x": 0.25, "y": 0.1, "width": 0.5, "height": 0.3},
        "lower": {"x": 0.25, "y": 0.4, "width": 0.5, "height": 0.3}
    }
    
    # Optimization settings
    ENABLE_XFORMERS = True
    ENABLE_CPU_OFFLOAD = True
    USE_FAST_SCHEDULER = True
    
    # Output settings
    OUTPUT_DIR = Path("outputs")
    SAVE_INTERMEDIATE = True
    SAVE_CONTROL_IMAGES = True
    
    # Fine-tuning configurations
    DREAMBOOTH_CONFIG = {
        "resolution": 512,
        "train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 5e-6,
        "lr_scheduler": "constant",
        "lr_warmup_steps": 0,
        "max_train_steps": 800,
        "checkpointing_steps": 500,
        "prior_loss_weight": 1.0,
        "seed": 1337,
    }
    
    LORA_CONFIG = {
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
        "lora_dropout": 0.1,
    }
    
    # Quality assessment thresholds
    QUALITY_THRESHOLDS = {
        "min_clip_score": 0.25,
        "max_fid_score": 50.0,
        "min_lpips_score": 0.1
    }
    
    @classmethod
    def get_product_size(cls, product_type, size):
        """Get size dimensions for a product"""
        if product_type.lower() == "tv":
            return cls.TV_SIZES.get(size, cls.TV_SIZES["55 inch"])
        elif product_type.lower() == "painting":
            return cls.PAINTING_SIZES.get(size, cls.PAINTING_SIZES["medium"])
        else:
            return {"width": 120, "height": 90}  # Default size
    
    @classmethod
    def get_prompt_template(cls, product_type, template_type="enhanced"):
        """Get prompt template for a product type"""
        return cls.PROMPTS.get(product_type.lower(), cls.PROMPTS["tv"])[template_type]
    
    @classmethod
    def get_wall_region(cls, region_name="center"):
        """Get wall region coordinates"""
        return cls.WALL_REGIONS.get(region_name, cls.WALL_REGIONS["center"])


class AdvancedConfig(Config):
    """Extended configuration for advanced features"""
    
    # Multi-model ensemble settings
    ENSEMBLE_MODELS = [
        "runwayml/stable-diffusion-v1-5",
        "stabilityai/stable-diffusion-2-1",
        "CompVis/stable-diffusion-v1-4"
    ]
    
    # Advanced ControlNet models
    ADVANCED_CONTROLNETS = {
        "depth": "lllyasviel/sd-controlnet-depth",
        "canny": "lllyasviel/sd-controlnet-canny",
        "openpose": "lllyasviel/sd-controlnet-openpose",
        "seg": "lllyasviel/sd-controlnet-seg",
        "normal": "lllyasviel/sd-controlnet-normal",
        "mlsd": "lllyasviel/sd-controlnet-mlsd"
    }
    
    # Multi-scale generation
    MULTI_SCALE_SIZES = [256, 512, 768, 1024]
    
    # Advanced prompt engineering
    STYLE_MODIFIERS = {
        "photorealistic": "photorealistic, ultra detailed, 8k resolution, professional photography",
        "artistic": "artistic rendering, painterly style, creative interpretation",
        "technical": "technical drawing, blueprint style, precise measurements",
        "luxury": "luxury interior, high-end design, premium materials, elegant"
    }
    
    # Batch processing settings
    BATCH_SIZE = 4
    MAX_CONCURRENT_GENERATIONS = 2
    
    # Performance monitoring
    ENABLE_PROFILING = False
    LOG_MEMORY_USAGE = True
    BENCHMARK_MODE = False
