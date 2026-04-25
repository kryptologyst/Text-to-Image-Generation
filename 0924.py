#!/usr/bin/env python3
"""
Project 924: Modern Text-to-Image Generation

This is a modernized, production-ready implementation of text-to-image generation
using state-of-the-art diffusion models. The project provides a comprehensive
framework for training, evaluating, and deploying text-to-image generation models.

Key Features:
- Multiple model support (Stable Diffusion v1.4/v2.1/XL, Kandinsky)
- Comprehensive evaluation metrics (CLIP Score, FID, KID, LPIPS)
- Safety features and content filtering
- Interactive web demo with Streamlit
- Production-ready code with type hints and testing

Usage:
    # Basic usage
    python 0924.py
    
    # With custom prompt
    python 0924.py --prompt "A beautiful sunset over mountains"
    
    # Run demo
    streamlit run demo/streamlit_app.py
    
    # Train model
    python scripts/train.py
    
    # Evaluate model
    python scripts/evaluate.py
"""

import argparse
import logging
from pathlib import Path
import torch
from PIL import Image

# Import our modern modules
from src.models import TextToImageGenerator, SafetyFilter
from src.utils import setup_logging, get_device, set_seed, format_prompt
from src.eval import TextToImageEvaluator
from src.viz import ImageVisualizer

logger = logging.getLogger(__name__)


def main():
    """Main function demonstrating modern text-to-image generation."""
    parser = argparse.ArgumentParser(description="Modern Text-to-Image Generation")
    parser.add_argument("--prompt", type=str, 
                       default="A futuristic city with flying cars and neon lights",
                       help="Text prompt for image generation")
    parser.add_argument("--model", type=str, 
                       default="CompVis/stable-diffusion-v-1-4-original",
                       help="Model ID to use")
    parser.add_argument("--output", type=str, default="generated_image.png",
                       help="Output image path")
    parser.add_argument("--steps", type=int, default=50,
                       help="Number of inference steps")
    parser.add_argument("--guidance-scale", type=float, default=7.5,
                       help="Guidance scale")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--safety-check", action="store_true",
                       help="Enable safety checking")
    parser.add_argument("--evaluate", action="store_true",
                       help="Run evaluation after generation")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging("INFO")
    logger.info("Starting modern text-to-image generation")
    
    # Set seed for reproducibility
    set_seed(args.seed)
    logger.info(f"Set random seed to {args.seed}")
    
    # Get device (CUDA > MPS > CPU)
    device = get_device()
    logger.info(f"Using device: {device}")
    
    try:
        # Initialize modern generator with safety features
        logger.info(f"Loading model: {args.model}")
        generator = TextToImageGenerator(
            model_id=args.model,
            device=device,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            safety_checker=args.safety_check,
            feature_extractor=True
        )
        
        # Safety check for prompt
        if args.safety_check:
            safety_filter = SafetyFilter(enable_nsfw_filter=True)
            safe_prompt, is_safe = safety_filter.filter_prompt(args.prompt)
            if not is_safe:
                logger.error("Prompt contains inappropriate content")
                return
            args.prompt = safe_prompt
        
        # Format and validate prompt
        formatted_prompt = format_prompt(args.prompt)
        logger.info(f"Generating image with prompt: '{formatted_prompt}'")
        
        # Generate image with modern parameters
        images = generator.generate(
            prompt=formatted_prompt,
            negative_prompt="blurry, low quality, distorted, ugly",
            height=512,
            width=512,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed
        )
        
        if images:
            # Save image
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            images[0].save(output_path)
            logger.info(f"Image saved to: {output_path}")
            
            # Display image info
            img = images[0]
            logger.info(f"Generated image: {img.size[0]}x{img.size[1]} pixels")
            
            # Optional evaluation
            if args.evaluate:
                logger.info("Running evaluation...")
                evaluator = TextToImageEvaluator(device=device)
                
                # Create reference image (placeholder for demo)
                reference_img = Image.new('RGB', (512, 512), (128, 128, 128))
                
                eval_results = evaluator.evaluate_generation(
                    generated_images=images,
                    prompts=[formatted_prompt],
                    reference_images=[reference_img]
                )
                
                logger.info("Evaluation Results:")
                logger.info(f"CLIP Score: {eval_results.get('clip_metrics', {}).get('clip_score', 0.0):.4f}")
                logger.info(f"Aesthetic Score: {eval_results.get('aesthetic', {}).get('aesthetic_score', 0.0):.4f}")
            
            # Create visualization
            viz = ImageVisualizer()
            viz.save_image_grid(
                images, 
                "assets/generated_samples.png",
                titles=[formatted_prompt],
                cols=1
            )
            logger.info("Visualization saved to assets/generated_samples.png")
            
        else:
            logger.error("Failed to generate image")
            
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        raise


if __name__ == "__main__":
    main()


# What This Modern Implementation Does:
# 
# 1. **Production-Ready Code**: Type hints, error handling, logging, and testing
# 2. **Device Management**: Automatic CUDA/MPS/CPU detection and fallback
# 3. **Safety Features**: NSFW filtering, prompt validation, safety checker
# 4. **Reproducibility**: Deterministic seeding and configuration management
# 5. **Evaluation**: Comprehensive metrics including CLIP Score, FID, KID
# 6. **Visualization**: Image grids, attention maps, and metric plots
# 7. **Extensibility**: Modular design supporting multiple models and tasks
# 8. **Documentation**: Comprehensive README, API docs, and examples
#
# This represents a significant upgrade from the original simple script to a
# full-featured, research-ready, and production-capable framework.

