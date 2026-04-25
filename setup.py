#!/usr/bin/env python3
"""Setup script for the text-to-image generation project."""

import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("🚀 Setting up Text-to-Image Generation Project")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Create necessary directories
    directories = [
        "data/images",
        "assets/evaluation",
        "checkpoints",
        "logs",
        "generated_images"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    # Install pre-commit hooks (optional)
    if run_command("pip install pre-commit", "Installing pre-commit"):
        run_command("pre-commit install", "Setting up pre-commit hooks")
    
    # Run basic tests
    if run_command("python -m pytest tests/ -v", "Running basic tests"):
        print("✅ All tests passed")
    else:
        print("⚠️ Some tests failed, but setup continues")
    
    # Generate synthetic dataset
    print("🔄 Generating synthetic dataset...")
    try:
        from src.data import SyntheticDatasetGenerator
        generator = SyntheticDatasetGenerator("data")
        generator.generate_dataset(num_samples=100, splits=["train", "val", "test"])
        print("✅ Synthetic dataset generated")
    except Exception as e:
        print(f"⚠️ Failed to generate synthetic dataset: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run the demo: streamlit run demo/streamlit_app.py")
    print("2. Generate an image: python 0924.py --prompt 'A beautiful landscape'")
    print("3. Train a model: python scripts/train.py")
    print("4. Evaluate a model: python scripts/evaluate.py")
    print("\nFor more information, see README.md")


if __name__ == "__main__":
    main()
