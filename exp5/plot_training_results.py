#!/usr/bin/env python3
"""
Training Results Visualization Script
Reads results.csv and generates training process charts for server environment (no GUI display)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

# Set matplotlib to non-interactive backend for server environment
matplotlib.use('Agg')

def plot_training_results(csv_path, output_dir):
    """
    Read training results CSV file and generate visualization charts
    
    Args:
        csv_path: path to results.csv file
        output_dir: output directory for charts
    """
    # Read training results
    df = pd.read_csv(csv_path)
    
    # Clean column names (remove extra spaces)
    df.columns = df.columns.str.strip()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set chart style
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    
    # Print available columns for debugging
    print("Available columns:", df.columns.tolist())
    
    # 1. Loss Function Charts
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('YOLOv5 Person Detection Model Training - Loss Functions', fontsize=16, fontweight='bold')
    
    # Train/Val Box Loss
    axes[0, 0].plot(df['epoch'], df['train/box_loss'], 'b-', label='Train Box Loss', linewidth=2)
    axes[0, 0].plot(df['epoch'], df['val/box_loss'], 'r-', label='Val Box Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Box Loss')
    axes[0, 0].set_title('Bounding Box Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Train/Val Object Loss
    axes[0, 1].plot(df['epoch'], df['train/obj_loss'], 'b-', label='Train Obj Loss', linewidth=2)
    axes[0, 1].plot(df['epoch'], df['val/obj_loss'], 'r-', label='Val Obj Loss', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Objectness Loss')
    axes[0, 1].set_title('Objectness Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Train/Val Class Loss
    axes[1, 0].plot(df['epoch'], df['train/cls_loss'], 'b-', label='Train Cls Loss', linewidth=2)
    axes[1, 0].plot(df['epoch'], df['val/cls_loss'], 'r-', label='Val Cls Loss', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Classification Loss')
    axes[1, 0].set_title('Classification Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[1, 1].plot(df['epoch'], df['x/lr0'], 'g-', label='Learning Rate', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_losses.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Performance Metrics Charts
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('YOLOv5 Person Detection Model Training - Performance Metrics', fontsize=16, fontweight='bold')
    
    # mAP@0.5 and mAP@0.5:0.95
    axes[0, 0].plot(df['epoch'], df['metrics/mAP_0.5'], 'purple', label='mAP@0.5', linewidth=3, marker='o', markersize=3)
    axes[0, 0].plot(df['epoch'], df['metrics/mAP_0.5:0.95'], 'orange', label='mAP@0.5:0.95', linewidth=3, marker='s', markersize=3)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('mAP')
    axes[0, 0].set_title('Mean Average Precision (mAP)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1])
    
    # Precision and Recall
    axes[0, 1].plot(df['epoch'], df['metrics/precision'], 'green', label='Precision', linewidth=3, marker='^', markersize=3)
    axes[0, 1].plot(df['epoch'], df['metrics/recall'], 'red', label='Recall', linewidth=3, marker='v', markersize=3)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Precision and Recall')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    # F1 Score
    f1_score = 2 * (df['metrics/precision'] * df['metrics/recall']) / (df['metrics/precision'] + df['metrics/recall'])
    axes[1, 0].plot(df['epoch'], f1_score, 'blue', label='F1 Score', linewidth=3, marker='d', markersize=3)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('F1 Score (Harmonic Mean of Precision & Recall)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])
    
    # Final Performance Summary
    final_metrics = df.iloc[-1]
    metrics_text = f"""Final Training Results (Epoch {int(final_metrics['epoch'])}):
    
mAP@0.5: {final_metrics['metrics/mAP_0.5']:.3f}
mAP@0.5:0.95: {final_metrics['metrics/mAP_0.5:0.95']:.3f}
Precision: {final_metrics['metrics/precision']:.3f}
Recall: {final_metrics['metrics/recall']:.3f}
F1 Score: {f1_score.iloc[-1]:.3f}

Box Loss: {final_metrics['val/box_loss']:.4f}
Obj Loss: {final_metrics['val/obj_loss']:.4f}
Cls Loss: {final_metrics['val/cls_loss']:.4f}"""
    
    axes[1, 1].text(0.05, 0.95, metrics_text, transform=axes[1, 1].transAxes, 
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    axes[1, 1].set_xlim([0, 1])
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Final Performance Summary')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Comprehensive Overview Chart
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('YOLOv5 Person Detection Training - Comprehensive Overview', fontsize=18, fontweight='bold')
    
    # All Loss Functions
    axes[0, 0].plot(df['epoch'], df['train/box_loss'], 'b-', label='Train Box', linewidth=2)
    axes[0, 0].plot(df['epoch'], df['val/box_loss'], 'r-', label='Val Box', linewidth=2)
    axes[0, 0].plot(df['epoch'], df['train/obj_loss'], 'b--', label='Train Obj', linewidth=2)
    axes[0, 0].plot(df['epoch'], df['val/obj_loss'], 'r--', label='Val Obj', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training & Validation Losses')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # All Performance Metrics
    axes[0, 1].plot(df['epoch'], df['metrics/mAP_0.5'], 'purple', label='mAP@0.5', linewidth=3)
    axes[0, 1].plot(df['epoch'], df['metrics/mAP_0.5:0.95'], 'orange', label='mAP@0.5:0.95', linewidth=3)
    axes[0, 1].plot(df['epoch'], df['metrics/precision'], 'green', label='Precision', linewidth=3)
    axes[0, 1].plot(df['epoch'], df['metrics/recall'], 'red', label='Recall', linewidth=3)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Performance Metrics')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    # Learning Rate & Memory Usage
    ax1 = axes[0, 2]
    ax2 = ax1.twinx()
    
    line1 = ax1.plot(df['epoch'], df['x/lr0'], 'g-', linewidth=2, label='Learning Rate')
    if 'x/lr1' in df.columns:
        line2 = ax2.plot(df['epoch'], df['x/lr1'], 'b-', linewidth=2, label='LR Group 1')
        ax2.set_ylabel('LR Group 1', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Learning Rate', color='g')
    ax1.tick_params(axis='y', labelcolor='g')
    ax1.set_title('Learning Rate Schedule')
    ax1.grid(True, alpha=0.3)
    
    # Training Progress Analysis
    improvement = df['metrics/mAP_0.5'].diff()
    axes[1, 0].bar(df['epoch'], improvement, alpha=0.7, color='skyblue', label='mAP Improvement')
    axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('mAP@0.5 Improvement')
    axes[1, 0].set_title('Training Progress (mAP@0.5 Improvement per Epoch)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Loss Convergence Analysis
    total_loss = df['val/box_loss'] + df['val/obj_loss'] + df['val/cls_loss']
    axes[1, 1].plot(df['epoch'], total_loss, 'red', linewidth=3, label='Total Validation Loss')
    axes[1, 1].plot(df['epoch'], total_loss.rolling(window=3).mean(), 'blue', linewidth=2, label='3-Epoch Moving Average')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Total Loss')
    axes[1, 1].set_title('Loss Convergence Analysis')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Model Performance Summary
    best_epoch = df.loc[df['metrics/mAP_0.5'].idxmax()]
    summary_text = f"""Training Summary:
    
Best Performance (Epoch {int(best_epoch['epoch'])}):
• mAP@0.5: {best_epoch['metrics/mAP_0.5']:.1%}
• mAP@0.5:0.95: {best_epoch['metrics/mAP_0.5:0.95']:.1%}
• Precision: {best_epoch['metrics/precision']:.1%}
• Recall: {best_epoch['metrics/recall']:.1%}

Final Performance (Epoch {int(final_metrics['epoch'])}):
• mAP@0.5: {final_metrics['metrics/mAP_0.5']:.1%}
• mAP@0.5:0.95: {final_metrics['metrics/mAP_0.5:0.95']:.1%}
• Precision: {final_metrics['metrics/precision']:.1%}
• Recall: {final_metrics['metrics/recall']:.1%}

Training Efficiency:
• Total Epochs: {len(df)}
• Peak mAP@0.5: {df['metrics/mAP_0.5'].max():.1%}
• Improvement: {(final_metrics['metrics/mAP_0.5'] - df['metrics/mAP_0.5'].iloc[0]):.1%}"""
    
    axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    axes[1, 2].set_xlim([0, 1])
    axes[1, 2].set_ylim([0, 1])
    axes[1, 2].axis('off')
    axes[1, 2].set_title('Training Summary')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Generate Training Report
    report_path = os.path.join(output_dir, 'training_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("YOLOv5 Person Detection Model Training Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("FINAL PERFORMANCE METRICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"mAP@0.5: {final_metrics['metrics/mAP_0.5']:.1%}\n")
        f.write(f"mAP@0.5:0.95: {final_metrics['metrics/mAP_0.5:0.95']:.1%}\n")
        f.write(f"Precision: {final_metrics['metrics/precision']:.1%}\n")
        f.write(f"Recall: {final_metrics['metrics/recall']:.1%}\n")
        f.write(f"F1 Score: {f1_score.iloc[-1]:.1%}\n\n")
        
        f.write("TRAINING PROCESS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total Epochs: {len(df)}\n")
        f.write(f"Best mAP@0.5: {df['metrics/mAP_0.5'].max():.1%} (Epoch {df.loc[df['metrics/mAP_0.5'].idxmax(), 'epoch']})\n")
        f.write(f"Final Learning Rate: {final_metrics['x/lr0']:.2e}\n")
        f.write(f"Final Box Loss: {final_metrics['val/box_loss']:.4f}\n")
        f.write(f"Final Object Loss: {final_metrics['val/obj_loss']:.4f}\n")
        f.write(f"Final Class Loss: {final_metrics['val/cls_loss']:.4f}\n\n")
        
        f.write("PERFORMANCE ANALYSIS:\n")
        f.write("-" * 30 + "\n")
        initial_map = df['metrics/mAP_0.5'].iloc[0]
        final_map = final_metrics['metrics/mAP_0.5']
        improvement = final_map - initial_map
        f.write(f"Initial mAP@0.5: {initial_map:.1%}\n")
        f.write(f"Final mAP@0.5: {final_map:.1%}\n")
        f.write(f"Total Improvement: {improvement:.1%}\n")
        f.write(f"Relative Improvement: {(improvement/initial_map)*100:.1f}%\n\n")
        
        f.write("EXPERIMENT REQUIREMENTS CHECK:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Target: mAP@0.5 > 90%\n")
        f.write(f"Achieved: {final_map:.1%}\n")
        f.write(f"Status: {'✅ PASSED' if final_map > 0.9 else '❌ FAILED'}\n")

if __name__ == "__main__":
    csv_path = "runs/train/person_detection_yolov5m4/results.csv"
    output_dir = "results/plots"
    
    if os.path.exists(csv_path):
        print("Generating training visualization charts...")
        plot_training_results(csv_path, output_dir)
        print("Charts generated successfully!")
        print(f"Output directory: {output_dir}")
        print("Generated files:")
        print("  - training_losses.png")
        print("  - training_metrics.png")
        print("  - training_overview.png")
        print("  - training_report.txt")
    else:
        print(f"Error: Results file not found at {csv_path}")
        print("Please make sure the training has been completed.") 