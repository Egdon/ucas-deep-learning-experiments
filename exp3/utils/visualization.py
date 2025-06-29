#!/usr/bin/env python3
"""
Training visualization utilities for Transformer poetry generation.
Supports plotting training loss, validation loss, learning rate, and perplexity curves.
"""

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional
import seaborn as sns
from datetime import datetime

class TransformerTrainingVisualizer:
    """Visualizer for Transformer training metrics with English labels"""
    
    def __init__(self, save_dir: str = "plots"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        # Color scheme
        self.colors = {
            'train_loss': '#2E86AB',
            'val_loss': '#A23B72', 
            'learning_rate': '#F18F01',
            'perplexity': '#C73E1D',
            'gradient_norm': '#52B788'
        }
    
    def plot_training_curves(self, 
                           metrics: Dict[str, List[float]], 
                           epochs: List[int],
                           title: str = "Transformer Poetry Generation Training",
                           save_name: str = None) -> str:
        """
        Plot comprehensive training curves
        
        Args:
            metrics: Dictionary with metric names as keys and values as lists
            epochs: List of epoch numbers
            title: Plot title
            save_name: Custom save filename
            
        Returns:
            Path to saved plot
        """
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(title, fontsize=18, fontweight='bold', y=0.95)
        
        # 1. Training and Validation Loss
        ax1 = axes[0, 0]
        if 'train_loss' in metrics:
            ax1.plot(epochs, metrics['train_loss'], 
                    color=self.colors['train_loss'], linewidth=2.5, 
                    label='Training Loss', marker='o', markersize=4)
        
        if 'val_loss' in metrics:
            ax1.plot(epochs, metrics['val_loss'], 
                    color=self.colors['val_loss'], linewidth=2.5,
                    label='Validation Loss', marker='s', markersize=4)
        
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')  # Log scale for better visualization
        
        # 2. Learning Rate Schedule
        ax2 = axes[0, 1]
        if 'learning_rate' in metrics:
            ax2.plot(epochs, metrics['learning_rate'], 
                    color=self.colors['learning_rate'], linewidth=2.5,
                    marker='d', markersize=4)
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('Learning Rate', fontsize=12)
            ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Learning Rate\nNot Available', 
                    ha='center', va='center', transform=ax2.transAxes,
                    fontsize=14, color='gray')
            ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        
        # 3. Perplexity
        ax3 = axes[1, 0]
        if 'perplexity' in metrics:
            ax3.plot(epochs, metrics['perplexity'], 
                    color=self.colors['perplexity'], linewidth=2.5,
                    marker='^', markersize=4, label='Perplexity')
        elif 'train_loss' in metrics and 'val_loss' in metrics:
            # Calculate perplexity from loss
            train_perplexity = [np.exp(loss) for loss in metrics['train_loss']]
            val_perplexity = [np.exp(loss) for loss in metrics['val_loss']]
            
            ax3.plot(epochs, train_perplexity, 
                    color=self.colors['train_loss'], linewidth=2.5,
                    marker='o', markersize=4, label='Train Perplexity')
            ax3.plot(epochs, val_perplexity, 
                    color=self.colors['val_loss'], linewidth=2.5,
                    marker='s', markersize=4, label='Val Perplexity')
            ax3.legend(fontsize=11)
        
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Perplexity', fontsize=12)
        ax3.set_title('Model Perplexity', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        
        # 4. Gradient Norm (if available)
        ax4 = axes[1, 1]
        if 'gradient_norm' in metrics:
            ax4.plot(epochs, metrics['gradient_norm'], 
                    color=self.colors['gradient_norm'], linewidth=2.5,
                    marker='*', markersize=5)
            ax4.set_xlabel('Epoch', fontsize=12)
            ax4.set_ylabel('Gradient Norm', fontsize=12)
            ax4.set_title('Gradient Norm', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Gradient Norm\nNot Available', 
                    ha='center', va='center', transform=ax4.transAxes,
                    fontsize=14, color='gray')
            ax4.set_title('Gradient Norm', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        if save_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_name = f"training_curves_{timestamp}.png"
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"📊 Training curves saved to: {save_path}")
        return str(save_path)
    
    def plot_loss_comparison(self, 
                           train_losses: List[float],
                           val_losses: List[float], 
                           epochs: List[int],
                           save_name: str = "loss_comparison.png") -> str:
        """
        Plot focused loss comparison
        """
        
        plt.figure(figsize=(12, 8))
        
        plt.plot(epochs, train_losses, 
                color=self.colors['train_loss'], linewidth=3,
                marker='o', markersize=6, label='Training Loss')
        
        plt.plot(epochs, val_losses, 
                color=self.colors['val_loss'], linewidth=3,
                marker='s', markersize=6, label='Validation Loss')
        
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14) 
        plt.title('Training vs Validation Loss', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"📊 Loss comparison saved to: {save_path}")
        return str(save_path)
    
    def plot_learning_rate_schedule(self,
                                  learning_rates: List[float],
                                  steps: List[int], 
                                  save_name: str = "learning_rate_schedule.png") -> str:
        """
        Plot detailed learning rate schedule
        """
        
        plt.figure(figsize=(12, 8))
        
        plt.plot(steps, learning_rates, 
                color=self.colors['learning_rate'], linewidth=3)
        
        plt.xlabel('Training Step', fontsize=14)
        plt.ylabel('Learning Rate', fontsize=14)
        plt.title('Learning Rate Schedule', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"📊 Learning rate schedule saved to: {save_path}")
        return str(save_path)
    
    def plot_perplexity_trend(self,
                            train_perplexity: List[float],
                            val_perplexity: List[float],
                            epochs: List[int],
                            save_name: str = "perplexity_trend.png") -> str:
        """
        Plot perplexity trend
        """
        
        plt.figure(figsize=(12, 8))
        
        plt.plot(epochs, train_perplexity, 
                color=self.colors['train_loss'], linewidth=3,
                marker='o', markersize=6, label='Training Perplexity')
        
        plt.plot(epochs, val_perplexity, 
                color=self.colors['val_loss'], linewidth=3,
                marker='s', markersize=6, label='Validation Perplexity')
        
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Perplexity', fontsize=14)
        plt.title('Model Perplexity Trend', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"📊 Perplexity trend saved to: {save_path}")
        return str(save_path)

def create_training_visualizer(save_dir: str = "plots") -> TransformerTrainingVisualizer:
    """Factory function to create visualizer"""
    return TransformerTrainingVisualizer(save_dir)

if __name__ == "__main__":
    # Test visualization
    print("Testing Transformer Training Visualizer...")
    
    # Create sample data
    epochs = list(range(1, 21))
    train_losses = [4.5 - 0.15*i + 0.05*np.random.randn() for i in epochs]
    val_losses = [4.7 - 0.12*i + 0.08*np.random.randn() for i in epochs]
    learning_rates = [0.001 * (0.95 ** i) for i in epochs]
    
    metrics = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'learning_rate': learning_rates
    }
    
    # Create visualizer and plot
    visualizer = create_training_visualizer("test_plots")
    visualizer.plot_training_curves(metrics, epochs, 
                                  title="Test Transformer Training")
    
    print("✅ Visualization test completed!") 