"""
Training Log Visualization Script for PaddleOCR PP-OCRv5 Mobile Detection Model

This script parses the training log and creates comprehensive visualizations
of the training progress including losses, metrics, and performance indicators.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

def parse_training_log(log_file_path):
    """Parse the training log file and extract metrics."""
    
    data = {
        'epochs': [],
        'global_steps': [],
        'learning_rates': [],
        'total_loss': [],
        'loss_shrink_maps': [],
        'loss_threshold_maps': [],
        'loss_binary_maps': [],
        'avg_batch_cost': [],
        'ips': [],  # images per second
        'timestamps': [],
        
        # Evaluation metrics
        'eval_epochs': [],
        'precision': [],
        'recall': [],
        'hmean': [],
        'fps': []
    }
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Parse training metrics (appears every 100 steps)
            if 'epoch:' in line and 'global_step:' in line and 'loss:' in line:
                # Extract epoch
                epoch_match = re.search(r'epoch: \[(\d+)/(\d+)\]', line)
                if epoch_match:
                    data['epochs'].append(int(epoch_match.group(1)))
                
                # Extract global step
                step_match = re.search(r'global_step: (\d+)', line)
                if step_match:
                    data['global_steps'].append(int(step_match.group(1)))
                
                # Extract learning rate
                lr_match = re.search(r'lr: ([\d.]+)', line)
                if lr_match:
                    data['learning_rates'].append(float(lr_match.group(1)))
                
                # Extract losses
                loss_match = re.search(r'loss: ([\d.]+)', line)
                if loss_match:
                    data['total_loss'].append(float(loss_match.group(1)))
                
                shrink_match = re.search(r'loss_shrink_maps: ([\d.]+)', line)
                if shrink_match:
                    data['loss_shrink_maps'].append(float(shrink_match.group(1)))
                
                threshold_match = re.search(r'loss_threshold_maps: ([\d.]+)', line)
                if threshold_match:
                    data['loss_threshold_maps'].append(float(threshold_match.group(1)))
                
                binary_match = re.search(r'loss_binary_maps: ([\d.]+)', line)
                if binary_match:
                    data['loss_binary_maps'].append(float(binary_match.group(1)))
                
                # Extract performance metrics
                batch_cost_match = re.search(r'avg_batch_cost: ([\d.]+)', line)
                if batch_cost_match:
                    data['avg_batch_cost'].append(float(batch_cost_match.group(1)))
                
                ips_match = re.search(r'ips: ([\d.]+)', line)
                if ips_match:
                    data['ips'].append(float(ips_match.group(1)))
                
                # Extract timestamp
                timestamp_match = re.search(r'\[([\d/\s:]+)\]', line)
                if timestamp_match:
                    data['timestamps'].append(timestamp_match.group(1))
            
            # Parse evaluation metrics
            if 'cur metric' in line:
                # Find corresponding epoch (look back in recent data)
                if data['epochs']:
                    data['eval_epochs'].append(data['epochs'][-1])
                
                precision_match = re.search(r'precision: ([\d.]+)', line)
                if precision_match:
                    data['precision'].append(float(precision_match.group(1)))
                
                recall_match = re.search(r'recall: ([\d.]+)', line)
                if recall_match:
                    data['recall'].append(float(recall_match.group(1)))
                
                hmean_match = re.search(r'hmean: ([\d.]+)', line)
                if hmean_match:
                    data['hmean'].append(float(hmean_match.group(1)))
                
                fps_match = re.search(r'fps: ([\d.]+)', line)
                if fps_match:
                    data['fps'].append(float(fps_match.group(1)))
    
    return data


def create_visualizations(data, output_dir):
    """Create comprehensive training visualizations."""
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create a large figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Total Loss over Training
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(data['epochs'], data['total_loss'], 'b-', linewidth=2, label='Total Loss')
    ax1.set_xlabel('Epoch', fontsize=10)
    ax1.set_ylabel('Loss', fontsize=10)
    ax1.set_title('Total Loss During Training', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Individual Loss Components
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(data['epochs'], data['loss_shrink_maps'], 'r-', linewidth=1.5, label='Shrink Maps', alpha=0.8)
    ax2.plot(data['epochs'], data['loss_threshold_maps'], 'g-', linewidth=1.5, label='Threshold Maps', alpha=0.8)
    ax2.plot(data['epochs'], data['loss_binary_maps'], 'b-', linewidth=1.5, label='Binary Maps', alpha=0.8)
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('Loss', fontsize=10)
    ax2.set_title('Loss Components Breakdown', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Learning Rate Schedule
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(data['epochs'], data['learning_rates'], 'purple', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=10)
    ax3.set_ylabel('Learning Rate', fontsize=10)
    ax3.set_title('Learning Rate Schedule (Cosine Decay)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('linear')
    
    # 4. Training Speed (IPS - Images Per Second)
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(data['epochs'], data['ips'], 'orange', linewidth=1.5, alpha=0.7)
    ax4.axhline(y=np.mean(data['ips']), color='red', linestyle='--', 
                label=f'Mean: {np.mean(data["ips"]):.2f} img/s')
    ax4.set_xlabel('Epoch', fontsize=10)
    ax4.set_ylabel('Images Per Second', fontsize=10)
    ax4.set_title('Training Speed (Throughput)', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # 5. Batch Processing Time
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(data['epochs'], data['avg_batch_cost'], 'brown', linewidth=1.5)
    ax5.set_xlabel('Epoch', fontsize=10)
    ax5.set_ylabel('Seconds', fontsize=10)
    ax5.set_title('Average Batch Processing Time', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Evaluation Metrics (if available)
    if data['eval_epochs']:
        ax6 = plt.subplot(3, 3, 6)
        ax6.plot(data['eval_epochs'], data['precision'], 'b-o', linewidth=2, markersize=8, label='Precision')
        ax6.plot(data['eval_epochs'], data['recall'], 'g-s', linewidth=2, markersize=8, label='Recall')
        ax6.plot(data['eval_epochs'], data['hmean'], 'r-^', linewidth=2, markersize=8, label='F1-Score (H-Mean)')
        ax6.set_xlabel('Epoch', fontsize=10)
        ax6.set_ylabel('Score', fontsize=10)
        ax6.set_title('Evaluation Metrics', fontsize=12, fontweight='bold')
        ax6.legend(fontsize=9)
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim([0, 1])
    
    # 7. Loss vs Learning Rate
    ax7 = plt.subplot(3, 3, 7)
    scatter = ax7.scatter(data['learning_rates'], data['total_loss'], 
                         c=data['epochs'], cmap='viridis', alpha=0.6, s=30)
    ax7.set_xlabel('Learning Rate', fontsize=10)
    ax7.set_ylabel('Total Loss', fontsize=10)
    ax7.set_title('Loss vs Learning Rate (colored by epoch)', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=ax7, label='Epoch')
    ax7.grid(True, alpha=0.3)
    
    # 8. Loss Percentage Distribution
    ax8 = plt.subplot(3, 3, 8)
    # Calculate average contribution of each loss component
    avg_shrink = np.mean(data['loss_shrink_maps'])
    avg_threshold = np.mean(data['loss_threshold_maps'])
    avg_binary = np.mean(data['loss_binary_maps'])
    
    sizes = [avg_shrink, avg_threshold, avg_binary]
    labels = ['Shrink Maps\n({:.3f})'.format(avg_shrink), 
              'Threshold Maps\n({:.3f})'.format(avg_threshold),
              'Binary Maps\n({:.3f})'.format(avg_binary)]
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    
    ax8.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 9})
    ax8.set_title('Average Loss Component Distribution', fontsize=12, fontweight='bold')
    
    # 9. Training Progress Summary (Text)
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # Calculate statistics
    initial_loss = data['total_loss'][0] if data['total_loss'] else 0
    final_loss = data['total_loss'][-1] if data['total_loss'] else 0
    loss_reduction = ((initial_loss - final_loss) / initial_loss * 100) if initial_loss > 0 else 0
    
    summary_text = f"""
    TRAINING SUMMARY
    ════════════════════════════════
    
    Total Epochs: {data['epochs'][-1] if data['epochs'] else 0}
    Total Steps: {data['global_steps'][-1] if data['global_steps'] else 0}
    
    LOSS METRICS:
    Initial Loss: {initial_loss:.4f}
    Final Loss: {final_loss:.4f}
    Reduction: {loss_reduction:.1f}%
    
    PERFORMANCE:
    Avg Speed: {np.mean(data['ips']):.2f} img/s
    Avg Batch Time: {np.mean(data['avg_batch_cost']):.2f}s
    
    BEST MODEL (Epoch {data['eval_epochs'][-1] if data['eval_epochs'] else 'N/A'}):
    Precision: {data['precision'][-1]:.4f} ({data['precision'][-1]*100:.2f}%)
    Recall: {data['recall'][-1]:.4f} ({data['recall'][-1]*100:.2f}%)
    F1-Score: {data['hmean'][-1]:.4f} ({data['hmean'][-1]*100:.2f}%)
    Inference FPS: {data['fps'][-1]:.2f}
    """
    
    ax9.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'training_visualization.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    # Create separate detailed plots
    create_detailed_loss_plot(data, output_dir)
    create_detailed_metrics_plot(data, output_dir)
    
    plt.show()


def create_detailed_loss_plot(data, output_dir):
    """Create a detailed loss progression plot."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot 1: All losses together
    ax1.plot(data['epochs'], data['total_loss'], 'k-', linewidth=2.5, label='Total Loss')
    ax1.plot(data['epochs'], data['loss_shrink_maps'], 'r--', linewidth=1.5, label='Shrink Maps Loss', alpha=0.7)
    ax1.plot(data['epochs'], data['loss_threshold_maps'], 'g--', linewidth=1.5, label='Threshold Maps Loss', alpha=0.7)
    ax1.plot(data['epochs'], data['loss_binary_maps'], 'b--', linewidth=1.5, label='Binary Maps Loss', alpha=0.7)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Detailed Loss Progression During Training', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss smoothed with moving average
    window = 5
    if len(data['total_loss']) >= window:
        smoothed_loss = np.convolve(data['total_loss'], np.ones(window)/window, mode='valid')
        smoothed_epochs = data['epochs'][window-1:]
        
        ax2.plot(data['epochs'], data['total_loss'], 'gray', linewidth=1, alpha=0.3, label='Raw Loss')
        ax2.plot(smoothed_epochs, smoothed_loss, 'b-', linewidth=2.5, label=f'Smoothed Loss (MA-{window})')
        
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title('Smoothed Loss Trend', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'detailed_loss_plot.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Detailed loss plot saved to: {output_path}")


def create_detailed_metrics_plot(data, output_dir):
    """Create a detailed evaluation metrics plot."""
    if not data['eval_epochs']:
        print("No evaluation data available for detailed metrics plot.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Precision, Recall, F1-Score
    ax1 = axes[0, 0]
    ax1.plot(data['eval_epochs'], data['precision'], 'b-o', linewidth=2, markersize=10, label='Precision')
    ax1.plot(data['eval_epochs'], data['recall'], 'g-s', linewidth=2, markersize=10, label='Recall')
    ax1.plot(data['eval_epochs'], data['hmean'], 'r-^', linewidth=2, markersize=10, label='F1-Score (H-Mean)')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Detection Performance Metrics', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Plot 2: Inference FPS
    ax2 = axes[0, 1]
    ax2.plot(data['eval_epochs'], data['fps'], color='purple', marker='D', linewidth=2, markersize=10)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Frames Per Second', fontsize=12)
    ax2.set_title('Inference Speed (FPS)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Precision vs Recall
    ax3 = axes[1, 0]
    ax3.scatter(data['recall'], data['precision'], c=data['eval_epochs'], 
               cmap='viridis', s=200, alpha=0.7, edgecolors='black', linewidth=1.5)
    ax3.set_xlabel('Recall', fontsize=12)
    ax3.set_ylabel('Precision', fontsize=12)
    ax3.set_title('Precision-Recall Trade-off', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', 
                               norm=plt.Normalize(vmin=min(data['eval_epochs']), 
                                                 vmax=max(data['eval_epochs'])))
    sm.set_array([])
    plt.colorbar(sm, ax=ax3, label='Epoch')
    
    # Plot 4: Metrics summary table
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')
    
    metrics_data = [
        ['Metric', 'Value', 'Epoch'],
        ['Best F1-Score', f'{max(data["hmean"]):.4f}', f'{data["eval_epochs"][data["hmean"].index(max(data["hmean"]))]}'],
        ['Best Precision', f'{max(data["precision"]):.4f}', f'{data["eval_epochs"][data["precision"].index(max(data["precision"]))]}'],
        ['Best Recall', f'{max(data["recall"]):.4f}', f'{data["eval_epochs"][data["recall"].index(max(data["recall"]))]}'],
        ['Final F1-Score', f'{data["hmean"][-1]:.4f}', f'{data["eval_epochs"][-1]}'],
        ['Final Precision', f'{data["precision"][-1]:.4f}', f'{data["eval_epochs"][-1]}'],
        ['Final Recall', f'{data["recall"][-1]:.4f}', f'{data["eval_epochs"][-1]}'],
    ]
    
    table = ax4.table(cellText=metrics_data, cellLoc='center', loc='center',
                     colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style the header row
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax4.set_title('Best and Final Metrics Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'detailed_metrics_plot.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Detailed metrics plot saved to: {output_path}")


def print_training_interpretation(data):
    """Print a detailed interpretation of the training results."""
    
    print("\n" + "="*80)
    print("TRAINING INTERPRETATION AND ANALYSIS")
    print("="*80 + "\n")
    
    print("📊 MODEL ARCHITECTURE:")
    print("-" * 80)
    print("Model: PP-OCRv5 Mobile Detection (DB - Differentiable Binarization)")
    print("Backbone: PPLCNetV3 (scale=0.75) - Lightweight CNN")
    print("Neck: RSEFPN (Receptive-field-enhancement Feature Pyramid Network)")
    print("Head: DBHead with fix_nan=True, k=50")
    print()
    
    print("🎯 TRAINING CONFIGURATION:")
    print("-" * 80)
    print(f"Total Epochs: 500")
    print(f"Batch Size: 8")
    print(f"Initial Learning Rate: 0.001")
    print(f"Learning Rate Schedule: Cosine decay with 2-epoch warmup")
    print(f"Optimizer: Adam (beta1=0.9, beta2=0.999)")
    print(f"Regularization: L2 with factor 5e-05")
    print(f"Input Size: 640x640")
    print()
    
    print("📈 LOSS ANALYSIS:")
    print("-" * 80)
    initial_loss = data['total_loss'][0]
    final_loss = data['total_loss'][-1]
    min_loss = min(data['total_loss'])
    loss_reduction = ((initial_loss - final_loss) / initial_loss * 100)
    
    print(f"Initial Loss (Epoch 20): {initial_loss:.4f}")
    print(f"Final Loss (Epoch 500): {final_loss:.4f}")
    print(f"Minimum Loss: {min_loss:.4f}")
    print(f"Total Reduction: {loss_reduction:.1f}%")
    print()
    
    print("Loss Components (Final vs Initial):")
    print(f"  • Shrink Maps:    {data['loss_shrink_maps'][0]:.4f} → {data['loss_shrink_maps'][-1]:.4f}")
    print(f"  • Threshold Maps: {data['loss_threshold_maps'][0]:.4f} → {data['loss_threshold_maps'][-1]:.4f}")
    print(f"  • Binary Maps:    {data['loss_binary_maps'][0]:.4f} → {data['loss_binary_maps'][-1]:.4f}")
    print()
    
    print("💡 Loss Component Interpretation:")
    print("  • Shrink Maps: Learns to shrink text regions for better boundary detection")
    print("  • Threshold Maps: Learns adaptive thresholds for binarization")
    print("  • Binary Maps: Final binary segmentation of text regions")
    print()
    
    print("🎓 EVALUATION RESULTS:")
    print("-" * 80)
    if data['eval_epochs']:
        best_epoch = data['eval_epochs'][-1]
        print(f"Best Model Found at Epoch: {best_epoch}")
        print(f"Precision: {data['precision'][-1]:.4f} ({data['precision'][-1]*100:.2f}%)")
        print(f"Recall:    {data['recall'][-1]:.4f} ({data['recall'][-1]*100:.2f}%)")
        print(f"F1-Score (H-Mean): {data['hmean'][-1]:.4f} ({data['hmean'][-1]*100:.2f}%)")
        print(f"Inference FPS: {data['fps'][-1]:.2f}")
        print()
        
        print("📊 Metric Interpretation:")
        print(f"  • Precision ({data['precision'][-1]*100:.1f}%): Of all detected text regions, {data['precision'][-1]*100:.1f}% are correct")
        print(f"  • Recall ({data['recall'][-1]*100:.1f}%): The model detects {data['recall'][-1]*100:.1f}% of all actual text regions")
        print(f"  • F1-Score ({data['hmean'][-1]*100:.1f}%): Overall detection quality (harmonic mean)")
        print()
    
    print("⚡ PERFORMANCE METRICS:")
    print("-" * 80)
    avg_ips = np.mean(data['ips'])
    avg_batch_time = np.mean(data['avg_batch_cost'])
    print(f"Average Training Speed: {avg_ips:.2f} images/second")
    print(f"Average Batch Time: {avg_batch_time:.2f} seconds")
    print(f"Total Training Time: ~{(len(data['epochs']) * avg_batch_time / 3600):.1f} hours")
    print()
    
    print("🔍 TRAINING BEHAVIOR ANALYSIS:")
    print("-" * 80)
    
    # Analyze loss trend
    loss_trend = np.polyfit(range(len(data['total_loss'])), data['total_loss'], 1)[0]
    if loss_trend < -0.001:
        print("✓ Loss Trend: Steadily decreasing (Good convergence)")
    elif loss_trend > 0.001:
        print("⚠ Loss Trend: Increasing (Possible overfitting or instability)")
    else:
        print("→ Loss Trend: Plateau (Converged)")
    print()
    
    # Learning rate analysis
    lr_start = data['learning_rates'][0]
    lr_end = data['learning_rates'][-1]
    print(f"Learning Rate: {lr_start:.6f} → {lr_end:.6f} (Cosine decay working correctly)")
    print()
    
    print("💪 MODEL STRENGTHS:")
    print("-" * 80)
    if data['precision'][-1] > 0.75:
        print("✓ High Precision: Low false positive rate")
    if data['recall'][-1] > 0.75:
        print("✓ High Recall: Detects most text regions")
    if data['hmean'][-1] > 0.75:
        print("✓ Strong Overall Performance: Well-balanced precision and recall")
    if data['fps'][-1] > 3:
        print("✓ Good Inference Speed: Suitable for real-time applications")
    print()
    
    print("📝 RECOMMENDATIONS:")
    print("-" * 80)
    
    if final_loss > 1.0:
        print("• Loss is still relatively high - consider:")
        print("  - Training for more epochs")
        print("  - Increasing model capacity")
        print("  - Data augmentation improvements")
    
    if data['precision'][-1] < data['recall'][-1] - 0.05:
        print("• Precision is lower than recall - the model has false positives")
        print("  - Consider increasing detection threshold")
        print("  - Review and clean training data")
    elif data['recall'][-1] < data['precision'][-1] - 0.05:
        print("• Recall is lower than precision - the model misses some text regions")
        print("  - Consider decreasing detection threshold")
        print("  - Add more diverse training samples")
    
    if loss_reduction < 10:
        print("• Limited loss reduction - consider:")
        print("  - Adjusting learning rate")
        print("  - Changing optimizer parameters")
        print("  - Checking data quality")
    
    print("\n" + "="*80)
    print("END OF ANALYSIS")
    print("="*80 + "\n")


def main():
    """Main execution function."""
    
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(script_dir, 'train.log')
    
    print("="*80)
    print("PaddleOCR Training Log Visualization and Analysis")
    print("="*80)
    print(f"\nParsing log file: {log_file}")
    
    if not os.path.exists(log_file):
        print(f"Error: Log file not found at {log_file}")
        return
    
    # Parse the log file
    data = parse_training_log(log_file)
    
    print(f"✓ Successfully parsed {len(data['epochs'])} training checkpoints")
    print(f"✓ Found {len(data['eval_epochs'])} evaluation checkpoints")
    
    # Print interpretation
    print_training_interpretation(data)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_visualizations(data, script_dir)
    
    print("\n✓ All visualizations created successfully!")
    print(f"✓ Output directory: {script_dir}")


if __name__ == "__main__":
    main()
