#!/usr/bin/env python3
"""Simple example script for text-to-image generation."""

import argparse
import logging
from pathlib import Path
from PIL import Image

from src.models import TextToImageGenerator, SafetyFilter
from src.utils import setup_logging, get_device, set_seed

logger = logging.getLogger(__name__)


def main():
    """Main function for simple text-to-image generation."""
    parser = argparse.ArgumentParser(description="Generate images from text prompts")
    parser.add_argument("--prompt", type=str, required=True,
                       help="Text prompt for image generation")
    parser.add_argument("--negative-prompt", type=str, default="blurry, low quality, distorted",
                       help="Negative prompt (what to avoid)")
    parser.add_argument("--model", type=str, default="CompVis/stable-diffusion-v-1-4-original",
                       help="Model ID to use")
    parser.add_argument("--output", type=str, default="generated_image.png",
                       help="Output image path")
    parser.add_argument("--steps", type=int, default=50,
                       help="Number of inference steps")
    parser.add_argument("--guidance-scale", type=float, default=7.5,
                       help="Guidance scale")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")
    parser.add_argument("--width", type=int, default=512,
                       help="Image width")
    parser.add_argument("--height", type=int, default=512,
                       help="Image height")
    parser.add_argument("--safety-check", action="store_true",
                       help="Enable safety checking")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging("INFO")
    
    # Safety check
    if args.safety_check:
        safety_filter = SafetyFilter(enable_nsfw_filter=True)
        safe_prompt, is_safe = safety_filter.filter_prompt(args.prompt)
        if not is_safe:
            logger.error("Prompt contains inappropriate content")
            return
        args.prompt = safe_prompt
    
    # Set seed if provided
    if args.seed is not None:
        set_seed(args.seed)
        logger.info(f"Set random seed to {args.seed}")
    
    # Initialize generator
    logger.info(f"Loading model: {args.model}")
    device = get_device()
    
    try:
        generator = TextToImageGenerator(
            model_id=args.model,
            device=device,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            safety_checker=args.safety_check
        )
        
        logger.info("Model loaded successfully")
        
        # Generate image
        logger.info(f"Generating image with prompt: '{args.prompt}'")
        
        images = generator.generate(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
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
            
        else:
            logger.error("Failed to generate image")
            
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        raise


if __name__ == "__main__":
    main()
