"""Data handling and preprocessing for text-to-image generation."""

import json
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ImageCaptionPair:
    """Data structure for image-caption pairs."""
    image_path: str
    caption: str
    metadata: Optional[Dict[str, Any]] = None


class TextToImageDataset(Dataset):
    """Dataset for text-to-image generation training/evaluation."""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        max_caption_length: int = 200,
        image_size: Tuple[int, int] = (512, 512),
        transform: Optional[Any] = None
    ):
        """Initialize dataset.
        
        Args:
            data_dir: Directory containing data
            split: Dataset split (train/val/test)
            max_caption_length: Maximum caption length
            image_size: Target image size
            transform: Image transformations
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_caption_length = max_caption_length
        self.image_size = image_size
        self.transform = transform
        
        # Load data
        self.data = self._load_data()
        
        logger.info(f"Loaded {len(self.data)} samples for {split} split")
    
    def _load_data(self) -> List[ImageCaptionPair]:
        """Load data from directory structure."""
        data = []
        
        # Look for annotations file
        annotations_file = self.data_dir / f"{self.split}_annotations.json"
        
        if annotations_file.exists():
            data = self._load_from_annotations(annotations_file)
        else:
            # Try to load from directory structure
            data = self._load_from_directory()
        
        return data
    
    def _load_from_annotations(self, annotations_file: Path) -> List[ImageCaptionPair]:
        """Load data from annotations JSON file."""
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        
        data = []
        for item in annotations:
            image_path = self.data_dir / "images" / item["image"]
            if image_path.exists():
                pair = ImageCaptionPair(
                    image_path=str(image_path),
                    caption=item["caption"],
                    metadata=item.get("metadata", {})
                )
                data.append(pair)
        
        return data
    
    def _load_from_directory(self) -> List[ImageCaptionPair]:
        """Load data from directory structure (images + captions.txt)."""
        data = []
        
        images_dir = self.data_dir / "images"
        captions_file = self.data_dir / f"{self.split}_captions.txt"
        
        if not images_dir.exists():
            logger.warning(f"Images directory not found: {images_dir}")
            return data
        
        # Load captions if available
        captions = {}
        if captions_file.exists():
            with open(captions_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t', 1)
                    if len(parts) == 2:
                        image_name, caption = parts
                        captions[image_name] = caption
        
        # Load images
        for image_file in images_dir.glob("*.jpg"):
            image_name = image_file.name
            caption = captions.get(image_name, f"Image {image_name}")
            
            pair = ImageCaptionPair(
                image_path=str(image_file),
                caption=caption
            )
            data.append(pair)
        
        return data
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item by index."""
        item = self.data[idx]
        
        # Load image
        try:
            image = Image.open(item.image_path).convert('RGB')
            image = image.resize(self.image_size, Image.Resampling.LANCZOS)
        except Exception as e:
            logger.error(f"Failed to load image {item.image_path}: {e}")
            # Return a blank image as fallback
            image = Image.new('RGB', self.image_size, (128, 128, 128))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Process caption
        caption = item.caption[:self.max_caption_length]
        
        return {
            "image": image,
            "caption": caption,
            "image_path": item.image_path,
            "metadata": item.metadata or {}
        }


class SyntheticDatasetGenerator:
    """Generate synthetic datasets for testing and demonstration."""
    
    def __init__(self, output_dir: Union[str, Path]):
        """Initialize synthetic dataset generator.
        
        Args:
            output_dir: Directory to save generated dataset
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sample prompts for synthetic data
        self.sample_prompts = [
            "A beautiful sunset over mountains",
            "A cute cat sitting on a windowsill",
            "A futuristic city with flying cars",
            "A peaceful lake with swans",
            "A colorful garden with flowers",
            "A cozy cabin in the woods",
            "A majestic eagle soaring in the sky",
            "A vintage car on a country road",
            "A snow-covered mountain peak",
            "A bustling marketplace",
            "A serene beach at sunset",
            "A modern skyscraper at night",
            "A field of sunflowers",
            "A stormy ocean with waves",
            "A peaceful forest path",
            "A hot air balloon in the sky",
            "A medieval castle on a hill",
            "A tropical island paradise",
            "A busy city street",
            "A starry night sky"
        ]
    
    def generate_dataset(
        self,
        num_samples: int = 100,
        image_size: Tuple[int, int] = (512, 512),
        splits: List[str] = None
    ) -> None:
        """Generate synthetic dataset.
        
        Args:
            num_samples: Number of samples to generate
            image_size: Size of generated images
            splits: Dataset splits to create
        """
        if splits is None:
            splits = ["train", "val", "test"]
        
        # Create directory structure
        images_dir = self.output_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        # Generate samples
        all_samples = []
        for i in range(num_samples):
            prompt = self.sample_prompts[i % len(self.sample_prompts)]
            
            # Create a simple synthetic image (colored rectangle)
            image = self._generate_synthetic_image(image_size, i)
            
            # Save image
            image_filename = f"sample_{i:04d}.jpg"
            image_path = images_dir / image_filename
            image.save(image_path)
            
            sample = {
                "image": image_filename,
                "caption": prompt,
                "metadata": {
                    "synthetic": True,
                    "sample_id": i
                }
            }
            all_samples.append(sample)
        
        # Split data
        train_size = int(0.7 * num_samples)
        val_size = int(0.15 * num_samples)
        
        train_samples = all_samples[:train_size]
        val_samples = all_samples[train_size:train_size + val_size]
        test_samples = all_samples[train_size + val_size:]
        
        # Save annotations
        splits_data = {
            "train": train_samples,
            "val": val_samples,
            "test": test_samples
        }
        
        for split in splits:
            if split in splits_data:
                annotations_file = self.output_dir / f"{split}_annotations.json"
                with open(annotations_file, 'w') as f:
                    json.dump(splits_data[split], f, indent=2)
        
        logger.info(f"Generated synthetic dataset with {num_samples} samples")
    
    def _generate_synthetic_image(
        self,
        size: Tuple[int, int],
        seed: int
    ) -> Image.Image:
        """Generate a simple synthetic image."""
        np.random.seed(seed)
        
        # Create a simple pattern
        width, height = size
        image_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        
        # Add some structure
        center_x, center_y = width // 2, height // 2
        
        # Create a gradient effect
        for y in range(height):
            for x in range(width):
                distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                intensity = max(0, 255 - distance // 2)
                
                image_array[y, x] = [
                    intensity,
                    (intensity + 50) % 256,
                    (intensity + 100) % 256
                ]
        
        return Image.fromarray(image_array)
    
    def create_captions_file(
        self,
        split: str = "train",
        format: str = "txt"
    ) -> None:
        """Create captions file in specified format."""
        annotations_file = self.output_dir / f"{split}_annotations.json"
        
        if not annotations_file.exists():
            logger.warning(f"Annotations file not found: {annotations_file}")
            return
        
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        
        if format == "txt":
            captions_file = self.output_dir / f"{split}_captions.txt"
            with open(captions_file, 'w') as f:
                for item in annotations:
                    f.write(f"{item['image']}\t{item['caption']}\n")
        
        logger.info(f"Created {format} captions file for {split} split")


def create_data_loader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    **kwargs
) -> DataLoader:
    """Create DataLoader for dataset.
    
    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        **kwargs: Additional DataLoader arguments
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        **kwargs
    )
