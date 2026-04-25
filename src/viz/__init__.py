"""Visualization utilities for text-to-image generation."""

import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns

logger = logging.getLogger(__name__)


class ImageVisualizer:
    """Visualization utilities for generated images."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """Initialize visualizer.
        
        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
        plt.style.use('default')
        sns.set_palette("husl")
    
    def create_image_grid(
        self,
        images: List[Image.Image],
        titles: Optional[List[str]] = None,
        cols: int = 4,
        figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """Create a grid of images.
        
        Args:
            images: List of PIL Images
            titles: Optional titles for each image
            cols: Number of columns
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if figsize is None:
            figsize = self.figsize
        
        rows = (len(images) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, (img, ax) in enumerate(zip(images, axes)):
            ax.imshow(img)
            ax.axis('off')
            
            if titles and i < len(titles):
                ax.set_title(titles[i], fontsize=10)
        
        # Hide empty subplots
        for i in range(len(images), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig
    
    def create_comparison_grid(
        self,
        original_images: List[Image.Image],
        generated_images: List[Image.Image],
        prompts: List[str],
        cols: int = 2
    ) -> plt.Figure:
        """Create comparison grid between original and generated images.
        
        Args:
            original_images: List of original PIL Images
            generated_images: List of generated PIL Images
            prompts: List of prompts
            cols: Number of columns
            
        Returns:
            Matplotlib figure
        """
        num_pairs = min(len(original_images), len(generated_images))
        rows = (num_pairs + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 8, rows * 4))
        
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_pairs):
            row = i // cols
            col = i % cols
            
            # Original image
            orig_ax = axes[row, col * 2]
            orig_ax.imshow(original_images[i])
            orig_ax.set_title(f"Original {i+1}", fontsize=10)
            orig_ax.axis('off')
            
            # Generated image
            gen_ax = axes[row, col * 2 + 1]
            gen_ax.imshow(generated_images[i])
            gen_ax.set_title(f"Generated {i+1}", fontsize=10)
            gen_ax.axis('off')
            
            # Add prompt as text
            if i < len(prompts):
                fig.text(0.5, 0.95 - i * 0.1, prompts[i], 
                        ha='center', fontsize=8, wrap=True)
        
        # Hide empty subplots
        for i in range(num_pairs, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col * 2].axis('off')
            axes[row, col * 2 + 1].axis('off')
        
        plt.tight_layout()
        return fig
    
    def create_attention_visualization(
        self,
        image: Image.Image,
        attention_map: np.ndarray,
        alpha: float = 0.6
    ) -> plt.Figure:
        """Create attention visualization overlay.
        
        Args:
            image: PIL Image
            attention_map: Attention map as numpy array
            alpha: Transparency of attention overlay
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        ax1.imshow(image)
        ax1.set_title("Original Image")
        ax1.axis('off')
        
        # Attention map
        im2 = ax2.imshow(attention_map, cmap='hot', interpolation='nearest')
        ax2.set_title("Attention Map")
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2)
        
        # Overlay
        ax3.imshow(image)
        ax3.imshow(attention_map, cmap='hot', alpha=alpha, interpolation='nearest')
        ax3.set_title("Attention Overlay")
        ax3.axis('off')
        
        plt.tight_layout()
        return fig
    
    def save_image_grid(
        self,
        images: List[Image.Image],
        output_path: Union[str, Path],
        titles: Optional[List[str]] = None,
        cols: int = 4
    ) -> None:
        """Save image grid to file.
        
        Args:
            images: List of PIL Images
            output_path: Output file path
            titles: Optional titles
            cols: Number of columns
        """
        fig = self.create_image_grid(images, titles, cols)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved image grid to {output_path}")


class MetricsVisualizer:
    """Visualization utilities for evaluation metrics."""
    
    def __init__(self):
        """Initialize metrics visualizer."""
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_metrics_comparison(
        self,
        model_results: Dict[str, Dict[str, Any]],
        metrics: List[str] = None
    ) -> plt.Figure:
        """Plot metrics comparison across models.
        
        Args:
            model_results: Dictionary mapping model names to results
            metrics: List of metrics to plot
            
        Returns:
            Matplotlib figure
        """
        if metrics is None:
            metrics = ["clip_score", "fid", "kid", "lpips_diversity", "aesthetic_score"]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if i >= len(axes):
                break
            
            ax = axes[i]
            model_names = []
            values = []
            
            for model_name, results in model_results.items():
                # Extract metric value from nested structure
                value = self._extract_metric_value(results, metric)
                if value is not None:
                    model_names.append(model_name)
                    values.append(value)
            
            if values:
                bars = ax.bar(model_names, values)
                ax.set_title(f"{metric.replace('_', ' ').title()}")
                ax.set_ylabel("Score")
                
                # Rotate x-axis labels if needed
                if len(model_names) > 3:
                    ax.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # Hide unused subplots
        for i in range(len(metrics), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig
    
    def _extract_metric_value(self, results: Dict[str, Any], metric: str) -> Optional[float]:
        """Extract metric value from nested results structure."""
        # Try different possible locations for the metric
        possible_locations = [
            results.get("clip_metrics", {}).get(metric),
            results.get("fid_kid", {}).get(metric),
            results.get("diversity", {}).get(metric),
            results.get("aesthetic", {}).get(metric),
            results.get(metric)
        ]
        
        for value in possible_locations:
            if value is not None:
                return float(value)
        
        return None
    
    def create_interactive_metrics_plot(
        self,
        model_results: Dict[str, Dict[str, Any]]
    ) -> go.Figure:
        """Create interactive metrics plot using Plotly.
        
        Args:
            model_results: Dictionary mapping model names to results
            
        Returns:
            Plotly figure
        """
        metrics = ["clip_score", "fid", "kid", "lpips_diversity", "aesthetic_score"]
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[m.replace('_', ' ').title() for m in metrics],
            specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}, None]]
        )
        
        for i, metric in enumerate(metrics):
            row = i // 3 + 1
            col = i % 3 + 1
            
            model_names = []
            values = []
            
            for model_name, results in model_results.items():
                value = self._extract_metric_value(results, metric)
                if value is not None:
                    model_names.append(model_name)
                    values.append(value)
            
            if values:
                fig.add_trace(
                    go.Bar(x=model_names, y=values, name=metric),
                    row=row, col=col
                )
        
        fig.update_layout(
            title="Model Performance Comparison",
            showlegend=False,
            height=600
        )
        
        return fig
    
    def plot_training_curves(
        self,
        training_data: Dict[str, List[float]],
        title: str = "Training Progress"
    ) -> plt.Figure:
        """Plot training curves.
        
        Args:
            training_data: Dictionary mapping metric names to lists of values
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for metric_name, values in training_data.items():
            ax.plot(values, label=metric_name)
        
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


class PromptVisualizer:
    """Visualization utilities for prompt analysis."""
    
    def __init__(self):
        """Initialize prompt visualizer."""
        pass
    
    def create_prompt_wordcloud(
        self,
        prompts: List[str],
        output_path: Optional[Union[str, Path]] = None
    ) -> plt.Figure:
        """Create word cloud from prompts.
        
        Args:
            prompts: List of text prompts
            output_path: Optional output path
            
        Returns:
            Matplotlib figure
        """
        try:
            from wordcloud import WordCloud
        except ImportError:
            logger.warning("wordcloud not installed, skipping word cloud generation")
            return None
        
        # Combine all prompts
        text = " ".join(prompts)
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            max_words=100,
            colormap='viridis'
        ).generate(text)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title("Prompt Word Cloud")
        
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def analyze_prompt_lengths(
        self,
        prompts: List[str]
    ) -> plt.Figure:
        """Analyze prompt length distribution.
        
        Args:
            prompts: List of text prompts
            
        Returns:
            Matplotlib figure
        """
        lengths = [len(prompt.split()) for prompt in prompts]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram
        ax1.hist(lengths, bins=20, alpha=0.7, edgecolor='black')
        ax1.set_xlabel("Prompt Length (words)")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Prompt Length Distribution")
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(lengths)
        ax2.set_ylabel("Prompt Length (words)")
        ax2.set_title("Prompt Length Box Plot")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
