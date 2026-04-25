"""Streamlit demo application for text-to-image generation."""

import streamlit as st
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import torch
from PIL import Image
import numpy as np
import time
import json
from datetime import datetime

from src.models import TextToImageGenerator, SafetyFilter
from src.utils import get_device, set_seed, load_config, ensure_dir
from src.viz import ImageVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Text-to-Image Generation",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_id: str, device: str) -> TextToImageGenerator:
    """Load and cache the text-to-image model."""
    try:
        model = TextToImageGenerator(
            model_id=model_id,
            device=torch.device(device),
            torch_dtype=torch.float16,
            safety_checker=True,
            feature_extractor=True
        )
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


def initialize_session_state():
    """Initialize session state variables."""
    if "generated_images" not in st.session_state:
        st.session_state.generated_images = []
    if "generation_history" not in st.session_state:
        st.session_state.generation_history = []
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False
    if "current_model_id" not in st.session_state:
        st.session_state.current_model_id = None


def load_configurations() -> Dict[str, Any]:
    """Load demo configurations."""
    try:
        config_path = Path("configs/demo/default.yaml")
        if config_path.exists():
            return load_config(config_path)
        else:
            # Default configuration
            return {
                "safety": {
                    "enable_nsfw_filter": True,
                    "enable_safety_checker": True,
                    "max_prompt_length": 200,
                    "blocked_words": ["explicit", "nsfw", "adult", "nude"]
                },
                "ui": {
                    "show_negative_prompt": True,
                    "show_guidance_scale": True,
                    "show_num_steps": True,
                    "show_seed": True,
                    "show_model_selection": True
                },
                "defaults": {
                    "prompt": "A beautiful landscape with mountains and a lake",
                    "negative_prompt": "blurry, low quality, distorted",
                    "guidance_scale": 7.5,
                    "num_inference_steps": 50,
                    "seed": None
                }
            }
    except Exception as e:
        logger.warning(f"Failed to load config: {e}")
        return {}


def main():
    """Main Streamlit application."""
    initialize_session_state()
    config = load_configurations()
    
    # Header
    st.markdown('<h1 class="main-header">🎨 Text-to-Image Generation</h1>', unsafe_allow_html=True)
    
    # Safety disclaimer
    st.markdown("""
    <div class="warning-box">
        <h4>⚠️ Safety Notice</h4>
        <p>This application generates images from text descriptions using AI. 
        Please use responsibly and avoid generating inappropriate content. 
        Generated images may not always match your expectations and should not be used for malicious purposes.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        if config.get("ui", {}).get("show_model_selection", True):
            st.subheader("Model Selection")
            model_options = {
                "Stable Diffusion v1.4": "CompVis/stable-diffusion-v-1-4-original",
                "Stable Diffusion v2.1": "stabilityai/stable-diffusion-2-1",
                "Stable Diffusion XL": "stabilityai/stable-diffusion-xl-base-1.0",
                "Kandinsky 2.1": "kandinsky-community/kandinsky-2-1"
            }
            
            selected_model_name = st.selectbox(
                "Choose Model:",
                list(model_options.keys()),
                index=0
            )
            model_id = model_options[selected_model_name]
        else:
            model_id = "CompVis/stable-diffusion-v-1-4-original"
        
        # Load model if changed
        if model_id != st.session_state.current_model_id:
            with st.spinner("Loading model..."):
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = load_model(model_id, device)
                if model:
                    st.session_state.model = model
                    st.session_state.model_loaded = True
                    st.session_state.current_model_id = model_id
                    st.success(f"✅ Model loaded: {selected_model_name}")
                else:
                    st.error("❌ Failed to load model")
                    st.session_state.model_loaded = False
        
        # Generation parameters
        st.subheader("🎛️ Generation Parameters")
        
        # Prompt
        default_prompt = config.get("defaults", {}).get("prompt", "A beautiful landscape with mountains and a lake")
        prompt = st.text_area(
            "Prompt:",
            value=default_prompt,
            height=100,
            help="Describe the image you want to generate"
        )
        
        # Negative prompt
        if config.get("ui", {}).get("show_negative_prompt", True):
            default_negative = config.get("defaults", {}).get("negative_prompt", "blurry, low quality, distorted")
            negative_prompt = st.text_area(
                "Negative Prompt:",
                value=default_negative,
                height=60,
                help="Describe what you don't want in the image"
            )
        else:
            negative_prompt = ""
        
        # Advanced parameters
        with st.expander("🔧 Advanced Parameters"):
            # Guidance scale
            if config.get("ui", {}).get("show_guidance_scale", True):
                guidance_scale = st.slider(
                    "Guidance Scale:",
                    min_value=1.0,
                    max_value=20.0,
                    value=config.get("defaults", {}).get("guidance_scale", 7.5),
                    step=0.5,
                    help="How closely to follow the prompt (higher = more adherence)"
                )
            else:
                guidance_scale = 7.5
            
            # Number of steps
            if config.get("ui", {}).get("show_num_steps", True):
                num_steps = st.slider(
                    "Inference Steps:",
                    min_value=10,
                    max_value=100,
                    value=config.get("defaults", {}).get("num_inference_steps", 50),
                    step=5,
                    help="Number of denoising steps (more = better quality, slower)"
                )
            else:
                num_steps = 50
            
            # Seed
            if config.get("ui", {}).get("show_seed", True):
                seed_option = st.radio(
                    "Seed:",
                    ["Random", "Fixed"],
                    help="Random seed for reproducibility"
                )
                if seed_option == "Fixed":
                    seed = st.number_input(
                        "Seed Value:",
                        min_value=0,
                        max_value=2**32-1,
                        value=42,
                        step=1
                    )
                else:
                    seed = None
            else:
                seed = None
            
            # Image size
            size_option = st.selectbox(
                "Image Size:",
                ["512x512", "768x768", "1024x1024"],
                index=0
            )
            width, height = map(int, size_option.split('x'))
        
        # Safety settings
        st.subheader("🛡️ Safety Settings")
        enable_safety = st.checkbox(
            "Enable Safety Checker",
            value=config.get("safety", {}).get("enable_safety_checker", True),
            help="Filter inappropriate content"
        )
        
        enable_nsfw_filter = st.checkbox(
            "Enable NSFW Filter",
            value=config.get("safety", {}).get("enable_nsfw_filter", True),
            help="Filter adult content"
        )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🎨 Image Generation")
        
        # Generate button
        if st.button("🚀 Generate Image", type="primary", use_container_width=True):
            if not st.session_state.model_loaded:
                st.error("Please select and load a model first")
            elif not prompt.strip():
                st.error("Please enter a prompt")
            else:
                # Safety check
                safety_filter = SafetyFilter(enable_nsfw_filter)
                safe_prompt, is_safe = safety_filter.filter_prompt(prompt)
                
                if not is_safe:
                    st.error("❌ Prompt contains inappropriate content. Please modify your prompt.")
                else:
                    # Generate image
                    with st.spinner("🎨 Generating image..."):
                        try:
                            start_time = time.time()
                            
                            # Set seed if provided
                            if seed is not None:
                                set_seed(seed)
                            
                            # Generate
                            images = st.session_state.model.generate(
                                prompt=safe_prompt,
                                negative_prompt=negative_prompt if negative_prompt else None,
                                height=height,
                                width=width,
                                num_inference_steps=num_steps,
                                guidance_scale=guidance_scale,
                                seed=seed
                            )
                            
                            generation_time = time.time() - start_time
                            
                            if images:
                                # Store generated image
                                st.session_state.generated_images = images
                                
                                # Add to history
                                history_entry = {
                                    "timestamp": datetime.now().isoformat(),
                                    "prompt": safe_prompt,
                                    "negative_prompt": negative_prompt,
                                    "guidance_scale": guidance_scale,
                                    "num_steps": num_steps,
                                    "seed": seed,
                                    "generation_time": generation_time,
                                    "model_id": model_id
                                }
                                st.session_state.generation_history.append(history_entry)
                                
                                st.success(f"✅ Image generated in {generation_time:.2f} seconds!")
                            else:
                                st.error("❌ Failed to generate image")
                        
                        except Exception as e:
                            st.error(f"❌ Generation failed: {e}")
                            logger.error(f"Generation error: {e}")
        
        # Display generated images
        if st.session_state.generated_images:
            st.subheader("🖼️ Generated Images")
            
            for i, image in enumerate(st.session_state.generated_images):
                st.image(image, caption=f"Generated Image {i+1}", use_column_width=True)
                
                # Download button
                img_bytes = image.tobytes()
                st.download_button(
                    label=f"📥 Download Image {i+1}",
                    data=img_bytes,
                    file_name=f"generated_image_{i+1}_{int(time.time())}.png",
                    mime="image/png"
                )
    
    with col2:
        st.header("📊 Generation Info")
        
        if st.session_state.generated_images:
            # Model info
            model_info = st.session_state.model.get_model_info()
            st.subheader("Model Information")
            st.write(f"**Model:** {model_id}")
            st.write(f"**Device:** {st.session_state.model.device}")
            st.write(f"**Supports Negative Prompt:** {model_info.get('supports_negative_prompt', 'Unknown')}")
            
            # Generation stats
            if st.session_state.generation_history:
                latest = st.session_state.generation_history[-1]
                st.subheader("Latest Generation")
                st.write(f"**Prompt:** {latest['prompt'][:100]}...")
                st.write(f"**Generation Time:** {latest['generation_time']:.2f}s")
                st.write(f"**Steps:** {latest['num_steps']}")
                st.write(f"**Guidance Scale:** {latest['guidance_scale']}")
                if latest['seed']:
                    st.write(f"**Seed:** {latest['seed']}")
        
        # History
        if st.session_state.generation_history:
            st.subheader("📝 Generation History")
            
            # Show last 5 generations
            recent_history = st.session_state.generation_history[-5:]
            for i, entry in enumerate(reversed(recent_history)):
                with st.expander(f"Generation {len(st.session_state.generation_history) - i}"):
                    st.write(f"**Time:** {entry['timestamp']}")
                    st.write(f"**Prompt:** {entry['prompt']}")
                    if entry['negative_prompt']:
                        st.write(f"**Negative:** {entry['negative_prompt']}")
                    st.write(f"**Time:** {entry['generation_time']:.2f}s")
            
            # Clear history button
            if st.button("🗑️ Clear History"):
                st.session_state.generation_history = []
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🎨 Text-to-Image Generation Demo | Built with Streamlit & Diffusers</p>
        <p><small>Use responsibly and follow community guidelines</small></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
