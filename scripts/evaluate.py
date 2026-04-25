#!/usr/bin/env python3
"""Evaluation script for text-to-image generation models."""

import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Any, List
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm

from src.models import TextToImageGenerator
from src.data import TextToImageDataset, create_data_loader
from src.eval import TextToImageEvaluator
from src.viz import ImageVisualizer, MetricsVisualizer
from src.utils import setup_logging, get_device, load_config, ensure_dir

logger = logging.getLogger(__name__)


class TextToImageEvaluator:
    """Comprehensive evaluator for text-to-image generation models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize evaluator.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.device = get_device()
        
        # Initialize components
        self.model = None
        self.evaluator = None
        self.image_viz = ImageVisualizer()
        self.metrics_viz = MetricsVisualizer()
        
        # Setup logging
        self.logger = setup_logging(config.get("log_level", "INFO"))
    
    def setup_model(self):
        """Setup the text-to-image model."""
        model_config = self.config["model"]
        
        self.model = TextToImageGenerator(
            model_id=model_config["model_id"],
            device=self.device,
            torch_dtype=getattr(torch, model_config.get("torch_dtype", "float16")),
            safety_checker=model_config.get("safety_checker", True),
            scheduler=model_config.get("scheduler")
        )
        
        self.logger.info(f"Initialized model: {model_config['model_id']}")
    
    def setup_evaluator(self):
        """Setup the evaluator."""
        eval_config = self.config["eval"]
        
        self.evaluator = TextToImageEvaluator(
            clip_model_name=eval_config.get("clip_model", "ViT-B/32"),
            device=self.device
        )
        
        self.logger.info("Initialized evaluator")
    
    def load_test_data(self) -> List[Dict[str, Any]]:
        """Load test data for evaluation."""
        data_dir = Path(self.config.get("data_dir", "data"))
        
        # Try to load test dataset
        test_dataset = TextToImageDataset(
            data_dir,
            split="test",
            max_caption_length=self.config.get("max_caption_length", 200),
            image_size=(512, 512)
        )
        
        if len(test_dataset) == 0:
            self.logger.warning("No test data found, creating synthetic test set")
            from src.data import SyntheticDatasetGenerator
            generator = SyntheticDatasetGenerator(data_dir)
            generator.generate_dataset(
                num_samples=self.config["eval"].get("generation", {}).get("num_samples", 100),
                splits=["test"]
            )
            test_dataset = TextToImageDataset(data_dir, split="test")
        
        # Convert to list for easier handling
        test_data = []
        for i in range(len(test_dataset)):
            item = test_dataset[i]
            test_data.append({
                "image": item["image"],
                "caption": item["caption"],
                "image_path": item["image_path"],
                "metadata": item["metadata"]
            })
        
        self.logger.info(f"Loaded {len(test_data)} test samples")
        return test_data
    
    def generate_images_for_evaluation(
        self,
        test_data: List[Dict[str, Any]]
    ) -> List[Image.Image]:
        """Generate images for evaluation."""
        eval_config = self.config["eval"]
        generation_config = eval_config.get("generation", {})
        
        generated_images = []
        
        self.logger.info("Generating images for evaluation...")
        
        for item in tqdm(test_data, desc="Generating images"):
            try:
                images = self.model.generate(
                    prompt=item["caption"],
                    num_inference_steps=generation_config.get("num_inference_steps", 50),
                    guidance_scale=generation_config.get("guidance_scale", 7.5),
                    height=512,
                    width=512
                )
                generated_images.extend(images)
            except Exception as e:
                self.logger.error(f"Failed to generate image for prompt '{item['caption']}': {e}")
                # Add placeholder image
                placeholder = Image.new('RGB', (512, 512), (128, 128, 128))
                generated_images.append(placeholder)
        
        self.logger.info(f"Generated {len(generated_images)} images")
        return generated_images
    
    def evaluate_model(
        self,
        test_data: List[Dict[str, Any]],
        generated_images: List[Image.Image]
    ) -> Dict[str, Any]:
        """Evaluate the model comprehensively."""
        self.logger.info("Starting comprehensive evaluation...")
        
        # Extract prompts and reference images
        prompts = [item["caption"] for item in test_data]
        reference_images = [item["image"] for item in test_data]
        
        # Ensure we have the same number of generated images as prompts
        if len(generated_images) != len(prompts):
            self.logger.warning(f"Mismatch: {len(generated_images)} generated vs {len(prompts)} prompts")
            # Truncate to match
            min_len = min(len(generated_images), len(prompts))
            generated_images = generated_images[:min_len]
            prompts = prompts[:min_len]
            reference_images = reference_images[:min_len]
        
        # Run evaluation
        eval_results = self.evaluator.evaluate_generation(
            generated_images=generated_images,
            prompts=prompts,
            reference_images=reference_images
        )
        
        # Add additional analysis
        eval_results["dataset_info"] = {
            "num_samples": len(test_data),
            "avg_prompt_length": np.mean([len(p.split()) for p in prompts]),
            "model_id": self.config["model"]["model_id"]
        }
        
        return eval_results
    
    def create_visualizations(
        self,
        test_data: List[Dict[str, Any]],
        generated_images: List[Image.Image],
        eval_results: Dict[str, Any]
    ):
        """Create evaluation visualizations."""
        self.logger.info("Creating visualizations...")
        
        # Ensure output directory exists
        output_dir = Path(self.config["eval"].get("results_dir", "assets/evaluation"))
        ensure_dir(output_dir)
        
        # Create image comparison grid
        if len(generated_images) >= 4:
            sample_size = min(8, len(generated_images))
            sample_indices = np.random.choice(len(generated_images), sample_size, replace=False)
            
            sample_generated = [generated_images[i] for i in sample_indices]
            sample_reference = [test_data[i]["image"] for i in sample_indices]
            sample_prompts = [test_data[i]["caption"] for i in sample_indices]
            
            # Create comparison grid
            fig = self.image_viz.create_comparison_grid(
                sample_reference,
                sample_generated,
                sample_prompts,
                cols=2
            )
            fig.savefig(output_dir / "comparison_grid.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        # Create generated images grid
        if len(generated_images) >= 4:
            sample_size = min(16, len(generated_images))
            sample_indices = np.random.choice(len(generated_images), sample_size, replace=False)
            
            sample_images = [generated_images[i] for i in sample_indices]
            sample_titles = [test_data[i]["caption"][:50] + "..." for i in sample_indices]
            
            fig = self.image_viz.create_image_grid(sample_images, sample_titles, cols=4)
            fig.savefig(output_dir / "generated_samples.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        # Create metrics visualization
        model_name = self.config["model"]["model_id"].split("/")[-1]
        model_results = {model_name: eval_results}
        
        fig = self.metrics_viz.plot_metrics_comparison(model_results)
        fig.savefig(output_dir / "metrics_comparison.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        self.logger.info(f"Visualizations saved to {output_dir}")
    
    def save_results(
        self,
        eval_results: Dict[str, Any],
        generated_images: List[Image.Image]
    ):
        """Save evaluation results."""
        output_dir = Path(self.config["eval"].get("results_dir", "assets/evaluation"))
        ensure_dir(output_dir)
        
        # Save results as JSON
        results_file = output_dir / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(eval_results, f, indent=2, default=str)
        
        # Save generated images if requested
        if self.config["eval"].get("save_images", True):
            images_dir = output_dir / "generated_images"
            ensure_dir(images_dir)
            
            for i, img in enumerate(generated_images):
                img.save(images_dir / f"generated_{i:04d}.png")
        
        self.logger.info(f"Results saved to {output_dir}")
    
    def create_leaderboard(self, eval_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a leaderboard entry."""
        model_name = self.config["model"]["model_id"].split("/")[-1]
        
        leaderboard_entry = {
            "model": model_name,
            "model_id": self.config["model"]["model_id"],
            "timestamp": str(pd.Timestamp.now()),
            "metrics": {
                "clip_score": eval_results.get("clip_metrics", {}).get("clip_score", 0.0),
                "fid": eval_results.get("fid_kid", {}).get("fid", float('inf')),
                "kid": eval_results.get("fid_kid", {}).get("kid", float('inf')),
                "lpips_diversity": eval_results.get("diversity", {}).get("lpips_diversity", 0.0),
                "aesthetic_score": eval_results.get("aesthetic", {}).get("aesthetic_score", 0.0)
            },
            "dataset_info": eval_results.get("dataset_info", {}),
            "config": self.config["model"]
        }
        
        return leaderboard_entry
    
    def evaluate(self):
        """Main evaluation function."""
        self.logger.info("Starting evaluation...")
        
        # Setup components
        self.setup_model()
        self.setup_evaluator()
        
        # Load test data
        test_data = self.load_test_data()
        
        # Generate images
        generated_images = self.generate_images_for_evaluation(test_data)
        
        # Evaluate model
        eval_results = self.evaluate_model(test_data, generated_images)
        
        # Create visualizations
        self.create_visualizations(test_data, generated_images, eval_results)
        
        # Save results
        self.save_results(eval_results, generated_images)
        
        # Create leaderboard entry
        leaderboard_entry = self.create_leaderboard(eval_results)
        
        # Print summary
        self.logger.info("Evaluation Summary:")
        self.logger.info(f"CLIP Score: {eval_results.get('clip_metrics', {}).get('clip_score', 0.0):.4f}")
        self.logger.info(f"FID: {eval_results.get('fid_kid', {}).get('fid', float('inf')):.4f}")
        self.logger.info(f"KID: {eval_results.get('fid_kid', {}).get('kid', float('inf')):.4f}")
        self.logger.info(f"LPIPS Diversity: {eval_results.get('diversity', {}).get('lpips_diversity', 0.0):.4f}")
        self.logger.info(f"Aesthetic Score: {eval_results.get('aesthetic', {}).get('aesthetic_score', 0.0):.4f}")
        
        return eval_results, leaderboard_entry


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate text-to-image generation model")
    parser.add_argument("--config", type=str, default="configs/eval/default.yaml",
                       help="Path to evaluation config file")
    parser.add_argument("--model-config", type=str, default="configs/model/default.yaml",
                       help="Path to model config file")
    parser.add_argument("--data-dir", type=str, default="data",
                       help="Path to data directory")
    parser.add_argument("--output-dir", type=str, default="assets/evaluation",
                       help="Path to output directory")
    
    args = parser.parse_args()
    
    # Load configurations
    eval_config = load_config(args.config)
    model_config = load_config(args.model_config)
    
    # Merge configurations
    config = {
        "eval": eval_config,
        "model": model_config,
        "data_dir": args.data_dir,
        "output_dir": args.output_dir
    }
    
    # Create evaluator and run evaluation
    evaluator = TextToImageEvaluator(config)
    eval_results, leaderboard_entry = evaluator.evaluate()
    
    # Save leaderboard entry
    leaderboard_file = Path(args.output_dir) / "leaderboard.json"
    with open(leaderboard_file, 'w') as f:
        json.dump(leaderboard_entry, f, indent=2)


if __name__ == "__main__":
    main()
