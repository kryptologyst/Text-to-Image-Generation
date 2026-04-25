# Text-to-Image Generation

Production-ready Multi-Modal AI project for text-to-image generation using state-of-the-art diffusion models. This project provides a comprehensive framework for training, evaluating, and deploying text-to-image generation models with safety features and extensive evaluation metrics.

## Features

- **Multiple Model Support**: Stable Diffusion v1.4/v2.1/XL, Kandinsky 2.1
- **Comprehensive Evaluation**: CLIP Score, FID, KID, LPIPS Diversity, Aesthetic Score
- **Safety Features**: NSFW filtering, safety checker, prompt validation
- **Interactive Demo**: Streamlit-based web interface
- **Production Ready**: Type hints, comprehensive testing, CI/CD
- **Extensible**: Modular design for easy customization

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Text-to-Image-Generation.git
cd Text-to-Image-Generation

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e ".[dev]"
```

### Basic Usage

```python
from src.models import TextToImageGenerator

# Initialize generator
generator = TextToImageGenerator(
    model_id="CompVis/stable-diffusion-v-1-4-original",
    device="cuda"  # or "cpu"
)

# Generate image
images = generator.generate(
    prompt="A beautiful landscape with mountains and a lake",
    negative_prompt="blurry, low quality",
    num_inference_steps=50,
    guidance_scale=7.5
)

# Save image
images[0].save("generated_image.png")
```

### Run Demo

```bash
# Start Streamlit demo
streamlit run demo/streamlit_app.py

# Or use the command line interface
python -m demo.streamlit_app
```

## Project Structure

```
text-to-image-generation/
├── src/                    # Source code
│   ├── models/            # Model implementations
│   ├── data/              # Data handling and preprocessing
│   ├── eval/              # Evaluation metrics
│   ├── viz/               # Visualization utilities
│   └── utils/             # Utility functions
├── configs/               # Configuration files
│   ├── model/            # Model configurations
│   ├── train/            # Training configurations
│   ├── eval/             # Evaluation configurations
│   └── demo/             # Demo configurations
├── scripts/              # Training and evaluation scripts
├── demo/                 # Demo applications
├── data/                 # Data directory
├── assets/               # Generated assets and results
├── tests/                # Test suite
└── notebooks/            # Jupyter notebooks
```

## Configuration

The project uses YAML configuration files for easy customization:

### Model Configuration (`configs/model/default.yaml`)

```yaml
model:
  name: "stable_diffusion_v1_4"
  model_id: "CompVis/stable-diffusion-v-1-4-original"
  scheduler: "PNDMScheduler"
  safety_checker: true
  
  generation:
    num_inference_steps: 50
    guidance_scale: 7.5
    height: 512
    width: 512
```

### Training Configuration (`configs/train/default.yaml`)

```yaml
train:
  batch_size: 4
  learning_rate: 1e-4
  num_epochs: 100
  mixed_precision: "fp16"
  save_every: 1000
```

## Training

### Prepare Data

The project supports multiple data formats:

1. **Annotations JSON**: `data/train_annotations.json`
2. **Captions TXT**: `data/train_captions.txt`
3. **Synthetic Data**: Automatically generated for testing

### Start Training

```bash
# Basic training
python scripts/train.py --config configs/train/default.yaml

# With custom data directory
python scripts/train.py --data-dir /path/to/data --use-wandb

# Resume from checkpoint
python scripts/train.py --checkpoint checkpoints/best_checkpoint.pt
```

### Training Features

- **Mixed Precision Training**: Automatic FP16 support
- **Safety Filtering**: Built-in prompt validation
- **Checkpointing**: Automatic model saving
- **Logging**: TensorBoard and Weights & Biases support
- **Validation**: Regular evaluation during training

## Evaluation

### Run Evaluation

```bash
# Evaluate model
python scripts/evaluate.py --config configs/eval/default.yaml

# Evaluate with custom model
python scripts/evaluate.py --model-config configs/model/custom.yaml
```

### Evaluation Metrics

- **CLIP Score**: Text-image alignment
- **FID (Fréchet Inception Distance)**: Image quality
- **KID (Kernel Inception Distance)**: Distribution similarity
- **LPIPS Diversity**: Generated image diversity
- **Aesthetic Score**: Visual quality assessment

### Leaderboard

The evaluation creates a comprehensive leaderboard comparing different models:

```json
{
  "model": "stable_diffusion_v1_4",
  "metrics": {
    "clip_score": 0.3124,
    "fid": 15.67,
    "kid": 0.0234,
    "lpips_diversity": 0.4567,
    "aesthetic_score": 0.7890
  }
}
```

## Demo Applications

### Streamlit Demo

The Streamlit demo provides an interactive web interface:

- **Model Selection**: Choose from multiple pre-trained models
- **Parameter Tuning**: Adjust generation parameters
- **Safety Controls**: Enable/disable safety features
- **Image Gallery**: View and download generated images
- **Generation History**: Track previous generations

### Features

- Real-time image generation
- Multiple model support
- Safety filtering
- Parameter customization
- Image download
- Generation history

## API Usage

### FastAPI Server

```bash
# Start API server
python -m demo.api_server

# Available endpoints:
# POST /generate - Generate images
# GET /models - List available models
# POST /evaluate - Evaluate model
```

### Example API Call

```python
import requests

response = requests.post("http://localhost:8000/generate", json={
    "prompt": "A beautiful sunset",
    "negative_prompt": "blurry, low quality",
    "guidance_scale": 7.5,
    "num_inference_steps": 50
})

image_data = response.json()["image"]
```

## Safety and Ethics

### Safety Features

- **NSFW Filtering**: Automatic detection and filtering
- **Prompt Validation**: Content policy enforcement
- **Safety Checker**: Built-in safety mechanisms
- **Watermarking**: Optional image watermarking

### Ethical Guidelines

- Use responsibly and ethically
- Respect copyright and intellectual property
- Avoid generating harmful or inappropriate content
- Consider the societal impact of generated content

### Disclaimers

- Generated images are for research and educational purposes
- Not intended for commercial use without proper licensing
- Users are responsible for compliance with applicable laws
- No guarantee of image quality or appropriateness

## Development

### Setup Development Environment

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Format code
black src/ scripts/ demo/
ruff check src/ scripts/ demo/
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test
pytest tests/test_models.py
```

### Code Quality

The project enforces high code quality standards:

- **Type Hints**: Full type annotation coverage
- **Documentation**: Google-style docstrings
- **Formatting**: Black code formatting
- **Linting**: Ruff static analysis
- **Testing**: Comprehensive test suite

## Performance Optimization

### Device Support

- **CUDA**: Automatic GPU acceleration
- **MPS**: Apple Silicon support
- **CPU**: Fallback CPU execution

### Memory Optimization

- **Mixed Precision**: FP16 training and inference
- **Memory Efficient Attention**: Reduced memory usage
- **Model Offloading**: CPU offloading for large models

### Batch Processing

```python
# Generate multiple images efficiently
images = generator.generate_batch(
    prompts=["prompt1", "prompt2", "prompt3"],
    negative_prompts=["neg1", "neg2", "neg3"],
    num_images_per_prompt=2
)
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Enable CPU offloading
   - Use smaller image sizes

2. **Model Loading Errors**
   - Check internet connection
   - Verify model ID
   - Clear Hugging Face cache

3. **Generation Quality Issues**
   - Increase inference steps
   - Adjust guidance scale
   - Improve prompt quality

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python scripts/train.py
```

## Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add type hints to all functions
- Write comprehensive tests
- Update documentation
- Ensure CI/CD passes

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{text_to_image_generation,
  title={Text-to-Image Generation: A Modern Multi-Modal AI Framework},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Text-to-Image-Generation}
}
```

## Acknowledgments

- Hugging Face for the Diffusers library
- Stability AI for Stable Diffusion models
- OpenAI for CLIP evaluation
- The open-source AI community

## Support

For questions and support:

- Create an issue on GitHub
- Check the documentation
- Review the troubleshooting guide
- Join our community discussions

---

**Disclaimer**: This project is for research and educational purposes. Generated images should be used responsibly and in compliance with applicable laws and ethical guidelines.
# Text-to-Image-Generation
