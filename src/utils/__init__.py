"""Utility functions for text-to-image generation project."""

import os
import random
import logging
from typing import Optional, Union, Dict, Any
import torch
import numpy as np
from pathlib import Path


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)


def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU).
    
    Returns:
        PyTorch device object
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    return device


def set_seed(seed: Optional[int] = None) -> int:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed. If None, generates a random seed.
        
    Returns:
        The seed that was set
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    return seed


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save config file
    """
    import yaml
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def format_prompt(prompt: str, max_length: int = 200) -> str:
    """Format and validate prompt.
    
    Args:
        prompt: Input prompt text
        max_length: Maximum prompt length
        
    Returns:
        Formatted prompt
    """
    if not prompt or not isinstance(prompt, str):
        raise ValueError("Prompt must be a non-empty string")
    
    # Clean and truncate
    prompt = prompt.strip()
    if len(prompt) > max_length:
        prompt = prompt[:max_length].rsplit(' ', 1)[0]  # Don't cut words
    
    return prompt


def validate_image_size(height: int, width: int) -> tuple[int, int]:
    """Validate and adjust image dimensions.
    
    Args:
        height: Image height
        width: Image width
        
    Returns:
        Validated (height, width) tuple
    """
    # Ensure dimensions are multiples of 8 (required by Stable Diffusion)
    height = ((height + 7) // 8) * 8
    width = ((width + 7) // 8) * 8
    
    # Ensure minimum size
    height = max(height, 64)
    width = max(width, 64)
    
    # Ensure maximum size (memory constraints)
    height = min(height, 1024)
    width = min(width, 1024)
    
    return height, width


def get_model_info(model_id: str) -> Dict[str, Any]:
    """Get information about a model.
    
    Args:
        model_id: Hugging Face model identifier
        
    Returns:
        Model information dictionary
    """
    return {
        "model_id": model_id,
        "is_sdxl": "xl" in model_id.lower(),
        "is_kandinsky": "kandinsky" in model_id.lower(),
        "supports_negative_prompt": True,
        "default_resolution": (1024, 1024) if "xl" in model_id.lower() else (512, 512)
    }
