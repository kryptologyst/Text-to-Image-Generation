"""Evaluation metrics for text-to-image generation."""

import logging
from typing import List, Dict, Any, Optional, Union
import torch
import numpy as np
from PIL import Image
import clip
from torchmetrics.image import FrechetInceptionDistance, KernelInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class TextToImageEvaluator:
    """Comprehensive evaluator for text-to-image generation models."""
    
    def __init__(
        self,
        clip_model_name: str = "ViT-B/32",
        device: Optional[torch.device] = None,
        fid_dims: int = 2048
    ):
        """Initialize evaluator.
        
        Args:
            clip_model_name: CLIP model name for evaluation
            device: PyTorch device
            fid_dims: FID feature dimensions
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fid_dims = fid_dims
        
        # Load CLIP model
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=self.device)
        
        # Initialize metrics
        self.fid = FrechetInceptionDistance(feature=2048, normalize=True).to(self.device)
        self.kid = KernelInceptionDistance(feature=2048, normalize=True).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(self.device)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"Initialized evaluator with CLIP {clip_model_name}")
    
    def compute_clip_score(
        self,
        images: List[Image.Image],
        texts: List[str]
    ) -> Dict[str, float]:
        """Compute CLIP score between images and texts.
        
        Args:
            images: List of PIL Images
            texts: List of text descriptions
            
        Returns:
            Dictionary with CLIP score metrics
        """
        if len(images) != len(texts):
            raise ValueError("Number of images and texts must match")
        
        # Preprocess images
        image_tensors = []
        for img in images:
            img_tensor = self.clip_preprocess(img).unsqueeze(0).to(self.device)
            image_tensors.append(img_tensor)
        
        image_tensors = torch.cat(image_tensors, dim=0)
        
        # Tokenize texts
        text_tokens = clip.tokenize(texts, truncate=True).to(self.device)
        
        # Get features
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_tensors)
            text_features = self.clip_model.encode_text(text_tokens)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Compute similarities
            similarities = torch.matmul(image_features, text_features.T)
            
            # CLIP score is the mean of diagonal elements
            clip_score = similarities.diag().mean().item()
            
            # Additional metrics
            max_similarity = similarities.max(dim=1)[0].mean().item()
            min_similarity = similarities.min(dim=1)[0].mean().item()
        
        return {
            "clip_score": clip_score,
            "max_similarity": max_similarity,
            "min_similarity": min_similarity,
            "similarity_std": similarities.diag().std().item()
        }
    
    def compute_fid_kid(
        self,
        real_images: List[Image.Image],
        generated_images: List[Image.Image]
    ) -> Dict[str, float]:
        """Compute FID and KID metrics.
        
        Args:
            real_images: List of real PIL Images
            generated_images: List of generated PIL Images
            
        Returns:
            Dictionary with FID and KID scores
        """
        # Convert PIL images to tensors
        def pil_to_tensor(images):
            tensors = []
            for img in images:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                tensor = transforms.ToTensor()(img)
                tensors.append(tensor)
            return torch.stack(tensors)
        
        real_tensors = pil_to_tensor(real_images).to(self.device)
        generated_tensors = pil_to_tensor(generated_images).to(self.device)
        
        # Update FID and KID
        self.fid.update(real_tensors, real=True)
        self.fid.update(generated_tensors, real=False)
        
        self.kid.update(real_tensors, real=True)
        self.kid.update(generated_tensors, real=False)
        
        # Compute metrics
        fid_score = self.fid.compute().item()
        kid_score = self.kid.compute().item()
        
        # Reset for next computation
        self.fid.reset()
        self.kid.reset()
        
        return {
            "fid": fid_score,
            "kid": kid_score
        }
    
    def compute_diversity_metrics(
        self,
        images: List[Image.Image]
    ) -> Dict[str, float]:
        """Compute diversity metrics for generated images.
        
        Args:
            images: List of generated PIL Images
            
        Returns:
            Dictionary with diversity metrics
        """
        if len(images) < 2:
            return {"lpips_diversity": 0.0, "ssim_diversity": 0.0}
        
        # Convert to tensors
        image_tensors = []
        for img in images:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            tensor = self.transform(img).unsqueeze(0).to(self.device)
            image_tensors.append(tensor)
        
        image_tensors = torch.cat(image_tensors, dim=0)
        
        # Compute LPIPS diversity
        lpips_scores = []
        for i in range(len(image_tensors)):
            for j in range(i + 1, len(image_tensors)):
                with torch.no_grad():
                    lpips_score = self.lpips(image_tensors[i:i+1], image_tensors[j:j+1])
                    lpips_scores.append(lpips_score.item())
        
        lpips_diversity = np.mean(lpips_scores) if lpips_scores else 0.0
        
        # Compute SSIM diversity (simplified)
        ssim_scores = []
        for i in range(len(image_tensors)):
            for j in range(i + 1, len(image_tensors)):
                # Simple SSIM approximation using MSE
                mse = torch.nn.functional.mse_loss(image_tensors[i], image_tensors[j])
                ssim_score = 1.0 / (1.0 + mse.item())
                ssim_scores.append(ssim_score)
        
        ssim_diversity = np.mean(ssim_scores) if ssim_scores else 0.0
        
        return {
            "lpips_diversity": lpips_diversity,
            "ssim_diversity": ssim_diversity,
            "num_pairs": len(lpips_scores)
        }
    
    def compute_aesthetic_score(
        self,
        images: List[Image.Image]
    ) -> Dict[str, float]:
        """Compute aesthetic quality scores.
        
        Args:
            images: List of PIL Images
            
        Returns:
            Dictionary with aesthetic scores
        """
        # This is a simplified aesthetic score based on image statistics
        # In practice, you would use a trained aesthetic predictor
        
        scores = []
        for img in images:
            # Convert to numpy array
            img_array = np.array(img)
            
            # Compute basic aesthetic indicators
            brightness = np.mean(img_array) / 255.0
            contrast = np.std(img_array) / 255.0
            
            # Simple aesthetic score (higher is better)
            aesthetic_score = brightness * contrast * 2.0
            scores.append(aesthetic_score)
        
        return {
            "aesthetic_score": np.mean(scores),
            "aesthetic_std": np.std(scores),
            "brightness": np.mean([np.mean(np.array(img)) / 255.0 for img in images]),
            "contrast": np.mean([np.std(np.array(img)) / 255.0 for img in images])
        }
    
    def evaluate_generation(
        self,
        generated_images: List[Image.Image],
        prompts: List[str],
        reference_images: Optional[List[Image.Image]] = None
    ) -> Dict[str, Any]:
        """Comprehensive evaluation of generated images.
        
        Args:
            generated_images: List of generated PIL Images
            prompts: List of text prompts
            reference_images: Optional reference images for FID/KID
            
        Returns:
            Dictionary with all evaluation metrics
        """
        results = {}
        
        # CLIP score
        if len(generated_images) == len(prompts):
            results["clip_metrics"] = self.compute_clip_score(generated_images, prompts)
        
        # FID/KID if reference images provided
        if reference_images:
            results["fid_kid"] = self.compute_fid_kid(reference_images, generated_images)
        
        # Diversity metrics
        results["diversity"] = self.compute_diversity_metrics(generated_images)
        
        # Aesthetic scores
        results["aesthetic"] = self.compute_aesthetic_score(generated_images)
        
        # Basic statistics
        results["basic_stats"] = {
            "num_images": len(generated_images),
            "avg_width": np.mean([img.width for img in generated_images]),
            "avg_height": np.mean([img.height for img in generated_images])
        }
        
        return results
    
    def create_leaderboard(
        self,
        model_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create a leaderboard from multiple model results.
        
        Args:
            model_results: Dictionary mapping model names to evaluation results
            
        Returns:
            Leaderboard with rankings
        """
        leaderboard = {}
        
        # Extract metrics for ranking
        metrics_to_rank = ["clip_score", "fid", "kid", "lpips_diversity", "aesthetic_score"]
        
        for metric in metrics_to_rank:
            if metric in ["fid", "kid"]:
                # Lower is better
                sorted_models = sorted(
                    model_results.items(),
                    key=lambda x: x[1].get("fid_kid", {}).get(metric, float('inf'))
                )
            else:
                # Higher is better
                sorted_models = sorted(
                    model_results.items(),
                    key=lambda x: x[1].get("clip_metrics", {}).get(metric, 0) +
                                 x[1].get("diversity", {}).get(metric, 0) +
                                 x[1].get("aesthetic", {}).get(metric, 0),
                    reverse=True
                )
            
            leaderboard[metric] = [
                {"model": name, "score": sorted_models[i][1]}
                for i, (name, _) in enumerate(sorted_models)
            ]
        
        return leaderboard
