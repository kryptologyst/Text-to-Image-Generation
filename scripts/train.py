#!/usr/bin/env python3
"""Training script for text-to-image generation models."""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Any
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from omegaconf import OmegaConf

from src.models import TextToImageGenerator, SafetyFilter
from src.data import TextToImageDataset, SyntheticDatasetGenerator, create_data_loader
from src.eval import TextToImageEvaluator
from src.viz import MetricsVisualizer
from src.utils import setup_logging, get_device, set_seed, load_config, save_config

logger = logging.getLogger(__name__)


class TextToImageTrainer:
    """Trainer for text-to-image generation models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.device = get_device()
        
        # Set seed for reproducibility
        self.seed = set_seed(config.get("seed", 42))
        
        # Initialize components
        self.model = None
        self.evaluator = None
        self.safety_filter = SafetyFilter()
        self.metrics_viz = MetricsVisualizer()
        
        # Training state
        self.current_epoch = 0
        self.best_score = float('inf')
        self.training_history = {
            "loss": [],
            "clip_score": [],
            "fid": []
        }
        
        # Setup logging
        self.logger = setup_logging(config.get("log_level", "INFO"))
        
        # Initialize wandb if configured
        if config.get("use_wandb", False):
            wandb.init(
                project="text-to-image-generation",
                config=config,
                name=f"run_{self.seed}"
            )
    
    def setup_data(self) -> tuple[DataLoader, DataLoader]:
        """Setup training and validation data loaders."""
        train_config = self.config["train"]
        
        # Create synthetic dataset if no real data exists
        data_dir = Path(self.config.get("data_dir", "data"))
        if not (data_dir / "train_annotations.json").exists():
            self.logger.info("Creating synthetic dataset...")
            generator = SyntheticDatasetGenerator(data_dir)
            generator.generate_dataset(
                num_samples=train_config.get("synthetic_samples", 1000),
                splits=["train", "val", "test"]
            )
        
        # Load datasets
        train_dataset = TextToImageDataset(
            data_dir,
            split="train",
            max_caption_length=self.config.get("max_caption_length", 200),
            image_size=(512, 512)
        )
        
        val_dataset = TextToImageDataset(
            data_dir,
            split="val",
            max_caption_length=self.config.get("max_caption_length", 200),
            image_size=(512, 512)
        )
        
        # Create data loaders
        train_loader = create_data_loader(
            train_dataset,
            batch_size=train_config["batch_size"],
            shuffle=True,
            num_workers=train_config.get("num_workers", 4)
        )
        
        val_loader = create_data_loader(
            val_dataset,
            batch_size=train_config.get("val_batch_size", 8),
            shuffle=False,
            num_workers=train_config.get("num_workers", 4)
        )
        
        self.logger.info(f"Loaded {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
        
        return train_loader, val_loader
    
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
        eval_config = self.config.get("eval", {})
        
        self.evaluator = TextToImageEvaluator(
            clip_model_name=eval_config.get("clip_model", "ViT-B/32"),
            device=self.device
        )
        
        self.logger.info("Initialized evaluator")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary with training metrics
        """
        self.model.pipeline.train()
        
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Get batch data
                images = batch["image"]
                captions = batch["caption"]
                
                # Filter prompts for safety
                safe_captions = []
                for caption in captions:
                    safe_caption, is_safe = self.safety_filter.filter_prompt(caption)
                    if is_safe:
                        safe_captions.append(safe_caption)
                    else:
                        safe_captions.append("A beautiful landscape")  # Fallback
                
                # Generate images
                generated_images = []
                for caption in safe_captions:
                    try:
                        images_batch = self.model.generate(
                            prompt=caption,
                            num_inference_steps=self.config["model"]["generation"]["num_inference_steps"],
                            guidance_scale=self.config["model"]["generation"]["guidance_scale"]
                        )
                        generated_images.extend(images_batch)
                    except Exception as e:
                        self.logger.warning(f"Generation failed for caption '{caption}': {e}")
                        # Add placeholder image
                        from PIL import Image
                        placeholder = Image.new('RGB', (512, 512), (128, 128, 128))
                        generated_images.append(placeholder)
                
                # Compute loss (simplified - in practice you'd use diffusion loss)
                # For demonstration, we'll use a simple reconstruction loss
                if len(generated_images) == len(images):
                    # Convert PIL images to tensors for loss computation
                    gen_tensors = []
                    orig_tensors = []
                    
                    for gen_img, orig_img in zip(generated_images, images):
                        # Simple tensor conversion
                        gen_tensor = torch.tensor(np.array(gen_img)).float() / 255.0
                        orig_tensor = torch.tensor(np.array(orig_img)).float() / 255.0
                        
                        gen_tensors.append(gen_tensor)
                        orig_tensors.append(orig_tensor)
                    
                    if gen_tensors and orig_tensors:
                        gen_batch = torch.stack(gen_tensors).to(self.device)
                        orig_batch = torch.stack(orig_tensors).to(self.device)
                        
                        # Simple MSE loss
                        loss = F.mse_loss(gen_batch, orig_batch)
                        
                        total_loss += loss.item()
                        num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({"loss": f"{total_loss/max(num_batches, 1):.4f}"})
                
                # Log to wandb
                if self.config.get("use_wandb", False) and batch_idx % 10 == 0:
                    wandb.log({
                        "train/loss": total_loss / max(num_batches, 1),
                        "train/batch": batch_idx,
                        "train/epoch": self.current_epoch
                    })
                
            except Exception as e:
                self.logger.error(f"Error in training batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        
        return {
            "loss": avg_loss,
            "num_batches": num_batches
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.pipeline.eval()
        
        all_generated_images = []
        all_prompts = []
        all_reference_images = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images = batch["image"]
                captions = batch["caption"]
                
                # Generate images
                for caption in captions:
                    try:
                        generated_images = self.model.generate(
                            prompt=caption,
                            num_inference_steps=self.config["model"]["generation"]["num_inference_steps"],
                            guidance_scale=self.config["model"]["generation"]["guidance_scale"]
                        )
                        all_generated_images.extend(generated_images)
                        all_prompts.extend([caption] * len(generated_images))
                    except Exception as e:
                        self.logger.warning(f"Validation generation failed: {e}")
                
                # Store reference images
                for img in images:
                    all_reference_images.append(img)
        
        # Compute evaluation metrics
        if all_generated_images and self.evaluator:
            eval_results = self.evaluator.evaluate_generation(
                generated_images=all_generated_images,
                prompts=all_prompts,
                reference_images=all_reference_images
            )
            
            return eval_results
        else:
            return {"clip_score": 0.0, "fid": float('inf')}
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch
            metrics: Current metrics
            is_best: Whether this is the best checkpoint
        """
        checkpoint_dir = Path(self.config.get("checkpoint_dir", "checkpoints"))
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            "epoch": epoch,
            "model_config": self.config["model"],
            "metrics": metrics,
            "seed": self.seed,
            "training_history": self.training_history
        }
        
        # Save regular checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = checkpoint_dir / "best_checkpoint.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best checkpoint at epoch {epoch}")
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        
        # Setup components
        self.setup_model()
        self.setup_evaluator()
        train_loader, val_loader = self.setup_data()
        
        train_config = self.config["train"]
        num_epochs = train_config["num_epochs"]
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            self.training_history["loss"].append(train_metrics["loss"])
            
            # Validation
            if epoch % train_config.get("val_every", 5) == 0:
                val_metrics = self.validate(val_loader)
                
                # Update training history
                clip_score = val_metrics.get("clip_metrics", {}).get("clip_score", 0.0)
                fid_score = val_metrics.get("fid_kid", {}).get("fid", float('inf'))
                
                self.training_history["clip_score"].append(clip_score)
                self.training_history["fid"].append(fid_score)
                
                # Check if best model
                is_best = fid_score < self.best_score
                if is_best:
                    self.best_score = fid_score
                
                # Save checkpoint
                if epoch % train_config.get("save_every", 10) == 0:
                    self.save_checkpoint(epoch, val_metrics, is_best)
                
                # Log metrics
                self.logger.info(
                    f"Epoch {epoch}: Loss={train_metrics['loss']:.4f}, "
                    f"CLIP Score={clip_score:.4f}, FID={fid_score:.4f}"
                )
                
                # Log to wandb
                if self.config.get("use_wandb", False):
                    wandb.log({
                        "epoch": epoch,
                        "train/loss": train_metrics["loss"],
                        "val/clip_score": clip_score,
                        "val/fid": fid_score,
                        "val/is_best": is_best
                    })
        
        self.logger.info("Training completed!")
        
        # Save final checkpoint
        final_metrics = self.validate(val_loader)
        self.save_checkpoint(num_epochs - 1, final_metrics, False)
        
        # Create training curves visualization
        if self.training_history["loss"]:
            fig = self.metrics_viz.plot_training_curves(self.training_history)
            fig.savefig("assets/training_curves.png", dpi=300, bbox_inches='tight')
            plt.close(fig)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train text-to-image generation model")
    parser.add_argument("--config", type=str, default="configs/train/default.yaml",
                       help="Path to training config file")
    parser.add_argument("--model-config", type=str, default="configs/model/default.yaml",
                       help="Path to model config file")
    parser.add_argument("--data-dir", type=str, default="data",
                       help="Path to data directory")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                       help="Path to checkpoint directory")
    parser.add_argument("--use-wandb", action="store_true",
                       help="Use Weights & Biases for logging")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Load configurations
    train_config = load_config(args.config)
    model_config = load_config(args.model_config)
    
    # Merge configurations
    config = {
        "train": train_config,
        "model": model_config,
        "data_dir": args.data_dir,
        "checkpoint_dir": args.checkpoint_dir,
        "use_wandb": args.use_wandb,
        "seed": args.seed
    }
    
    # Create trainer and start training
    trainer = TextToImageTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
