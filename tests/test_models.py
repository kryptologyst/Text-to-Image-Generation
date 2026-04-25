"""Test suite for text-to-image generation project."""

import pytest
import torch
from PIL import Image
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.models import TextToImageGenerator, SafetyFilter
from src.data import TextToImageDataset, SyntheticDatasetGenerator
from src.eval import TextToImageEvaluator
from src.utils import get_device, set_seed, format_prompt, validate_image_size


class TestTextToImageGenerator:
    """Test cases for TextToImageGenerator."""
    
    def test_initialization(self):
        """Test model initialization."""
        generator = TextToImageGenerator(
            model_id="CompVis/stable-diffusion-v-1-4-original",
            device="cpu",  # Use CPU for testing
            torch_dtype=torch.float32
        )
        assert generator.model_id == "CompVis/stable-diffusion-v-1-4-original"
        assert generator.device.type == "cpu"
    
    def test_prompt_formatting(self):
        """Test prompt formatting."""
        generator = TextToImageGenerator(
            model_id="CompVis/stable-diffusion-v-1-4-original",
            device="cpu",
            torch_dtype=torch.float32
        )
        
        # Test valid prompt
        prompt = "A beautiful landscape"
        formatted = format_prompt(prompt)
        assert formatted == prompt
        
        # Test long prompt truncation
        long_prompt = "A " * 300  # Very long prompt
        formatted = format_prompt(long_prompt, max_length=200)
        assert len(formatted) <= 200
    
    def test_image_size_validation(self):
        """Test image size validation."""
        # Test normal sizes
        height, width = validate_image_size(512, 512)
        assert height == 512
        assert width == 512
        
        # Test sizes that need adjustment
        height, width = validate_image_size(513, 513)  # Not multiple of 8
        assert height == 512  # Should be adjusted to multiple of 8
        assert width == 512
        
        # Test minimum size
        height, width = validate_image_size(32, 32)
        assert height == 64  # Minimum size
        assert width == 64
        
        # Test maximum size
        height, width = validate_image_size(2048, 2048)
        assert height == 1024  # Maximum size
        assert width == 1024


class TestSafetyFilter:
    """Test cases for SafetyFilter."""
    
    def test_initialization(self):
        """Test safety filter initialization."""
        filter_obj = SafetyFilter(enable_nsfw_filter=True)
        assert filter_obj.enable_nsfw_filter is True
        assert len(filter_obj.blocked_words) > 0
    
    def test_prompt_filtering(self):
        """Test prompt filtering."""
        filter_obj = SafetyFilter(enable_nsfw_filter=True)
        
        # Test safe prompt
        safe_prompt = "A beautiful landscape"
        filtered_prompt, is_safe = filter_obj.filter_prompt(safe_prompt)
        assert is_safe is True
        assert filtered_prompt == safe_prompt
        
        # Test unsafe prompt
        unsafe_prompt = "explicit content"
        filtered_prompt, is_safe = filter_obj.filter_prompt(unsafe_prompt)
        assert is_safe is False
        assert filtered_prompt == ""
    
    def test_image_safety_check(self):
        """Test image safety check."""
        filter_obj = SafetyFilter()
        
        # Create a test image
        test_image = Image.new('RGB', (100, 100), (255, 255, 255))
        
        # Test safety check (placeholder implementation)
        is_safe = filter_obj.check_image_safety(test_image)
        assert isinstance(is_safe, bool)


class TestSyntheticDatasetGenerator:
    """Test cases for SyntheticDatasetGenerator."""
    
    def test_initialization(self):
        """Test dataset generator initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = SyntheticDatasetGenerator(temp_dir)
            assert Path(temp_dir).exists()
    
    def test_dataset_generation(self):
        """Test synthetic dataset generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = SyntheticDatasetGenerator(temp_dir)
            
            # Generate small dataset
            generator.generate_dataset(
                num_samples=10,
                splits=["train", "val"]
            )
            
            # Check if files were created
            assert (Path(temp_dir) / "images").exists()
            assert (Path(temp_dir) / "train_annotations.json").exists()
            assert (Path(temp_dir) / "val_annotations.json").exists()
            
            # Check if images were generated
            images_dir = Path(temp_dir) / "images"
            image_files = list(images_dir.glob("*.jpg"))
            assert len(image_files) == 10


class TestTextToImageDataset:
    """Test cases for TextToImageDataset."""
    
    def test_dataset_loading(self):
        """Test dataset loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create synthetic dataset first
            generator = SyntheticDatasetGenerator(temp_dir)
            generator.generate_dataset(num_samples=5, splits=["train"])
            
            # Load dataset
            dataset = TextToImageDataset(temp_dir, split="train")
            assert len(dataset) == 5
            
            # Test getting an item
            item = dataset[0]
            assert "image" in item
            assert "caption" in item
            assert "image_path" in item
            assert isinstance(item["image"], Image.Image)
            assert isinstance(item["caption"], str)


class TestTextToImageEvaluator:
    """Test cases for TextToImageEvaluator."""
    
    def test_initialization(self):
        """Test evaluator initialization."""
        evaluator = TextToImageEvaluator(device="cpu")
        assert evaluator.device.type == "cpu"
        assert evaluator.clip_model is not None
    
    def test_clip_score_computation(self):
        """Test CLIP score computation."""
        evaluator = TextToImageEvaluator(device="cpu")
        
        # Create test images and prompts
        images = [Image.new('RGB', (224, 224), (255, 0, 0)) for _ in range(2)]
        prompts = ["A red image", "Another red image"]
        
        # Compute CLIP score
        results = evaluator.compute_clip_score(images, prompts)
        
        assert "clip_score" in results
        assert "max_similarity" in results
        assert "min_similarity" in results
        assert isinstance(results["clip_score"], float)
    
    def test_diversity_metrics(self):
        """Test diversity metrics computation."""
        evaluator = TextToImageEvaluator(device="cpu")
        
        # Create test images
        images = [Image.new('RGB', (224, 224), (i*50, 100, 200)) for i in range(3)]
        
        # Compute diversity metrics
        results = evaluator.compute_diversity_metrics(images)
        
        assert "lpips_diversity" in results
        assert "ssim_diversity" in results
        assert isinstance(results["lpips_diversity"], float)
        assert isinstance(results["ssim_diversity"], float)


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ["cpu", "cuda", "mps"]
    
    def test_set_seed(self):
        """Test seed setting."""
        seed = set_seed(42)
        assert seed == 42
        
        # Test random seed generation
        seed = set_seed(None)
        assert isinstance(seed, int)
        assert 0 <= seed <= 2**32 - 1
    
    def test_format_prompt(self):
        """Test prompt formatting."""
        # Test normal prompt
        prompt = "A beautiful landscape"
        formatted = format_prompt(prompt)
        assert formatted == prompt
        
        # Test empty prompt
        with pytest.raises(ValueError):
            format_prompt("")
        
        # Test non-string prompt
        with pytest.raises(ValueError):
            format_prompt(None)
        
        # Test long prompt truncation
        long_prompt = "A " * 300
        formatted = format_prompt(long_prompt, max_length=200)
        assert len(formatted) <= 200
        assert not formatted.endswith(" ")  # Should not end with space
    
    def test_validate_image_size(self):
        """Test image size validation."""
        # Test normal case
        h, w = validate_image_size(512, 512)
        assert h == 512
        assert w == 512
        
        # Test adjustment to multiple of 8
        h, w = validate_image_size(513, 513)
        assert h == 512
        assert w == 512
        
        # Test minimum size enforcement
        h, w = validate_image_size(32, 32)
        assert h == 64
        assert w == 64
        
        # Test maximum size enforcement
        h, w = validate_image_size(2048, 2048)
        assert h == 1024
        assert w == 1024


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_generation(self):
        """Test end-to-end image generation."""
        # This test requires actual model loading, so we'll mock it
        # In a real test environment, you might want to use a smaller model
        
        # Test with CPU and minimal parameters
        generator = TextToImageGenerator(
            model_id="CompVis/stable-diffusion-v-1-4-original",
            device="cpu",
            torch_dtype=torch.float32
        )
        
        # Test generation parameters validation
        prompt = "A simple test image"
        height, width = validate_image_size(256, 256)
        
        assert height == 256
        assert width == 256
        
        # Note: Actual generation would require model loading
        # which might be too slow for unit tests
    
    def test_safety_integration(self):
        """Test safety filter integration."""
        filter_obj = SafetyFilter(enable_nsfw_filter=True)
        
        # Test safe prompt
        safe_prompt = "A beautiful landscape with mountains"
        filtered, is_safe = filter_obj.filter_prompt(safe_prompt)
        assert is_safe is True
        
        # Test unsafe prompt
        unsafe_prompt = "explicit adult content"
        filtered, is_safe = filter_obj.filter_prompt(unsafe_prompt)
        assert is_safe is False


if __name__ == "__main__":
    pytest.main([__file__])
