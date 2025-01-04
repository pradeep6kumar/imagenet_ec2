# ImageNet-1K Classification with ResNet50

## Project Overview
This project implements a ResNet50-based classifier for the ImageNet-1K dataset, using PyTorch with advanced training techniques including SWA (Stochastic Weight Averaging) and mixed-precision training.

## Architecture
- **Base Model**: ResNet50
- **Dataset**: ImageNet-1K (1000 classes)
- **Framework**: PyTorch
- **Training Hardware**: Single GPU (CUDA)

## Key Features
- Stochastic Weight Averaging (SWA)
- Mixed Precision Training (AMP)
- OneCycleLR Learning Rate Scheduling
- Checkpoint Management
- Automated Learning Rate Finding
- GPU Memory Optimization

## Best Performance
- **Maximum Validation Accuracy**: 59.21%
- **Best Training Accuracy**: 51.52%
- **Final Loss**: 1.6899

## Training Progress Summary

### Initial Phase (Epochs 1-10)

Epoch 1: 13.90% validation accuracy

Epoch 5: 37.19% validation accuracy

Epoch 10: 44.15% validation accuracy

### Mid Training (Epochs 11-20)

Epoch 15: 46.53% validation accuracy

Epoch 20: 48.01% validation accuracy

### Final Phase with SWA (Epochs 32+)

After SWA activation: 59.21% validation accuracy

## Training Configuration

```python
{

'batch_size': 256,

'learning_rate': 0.0005,

'epochs': 100,

'weight_decay': 1e-4,

'warmup_epochs': 10,

'grad_clip_value': 1.0

}
```



## Training Metrics
- **Training Time per Epoch**: ~1.27 hours
- **Samples per Second**: ~282 images/s
- **GPU Memory Usage**: ~11.7GB/15.36GB
- **Training Stability**: Consistent improvement until plateau

## Key Training Phases
1. **Initial Learning (Epochs 1-10)**
   - Rapid improvement from 13.90% to 44.15%
   - Stable learning rate at 0.00025

2. **Intermediate Learning (Epochs 11-31)**
   - Gradual improvement
   - Consistent validation accuracy gains

3. **SWA Phase (Epochs 32+)**
   - Activated at epoch 32
   - Significant improvement in model stability
   - Peak performance at 59.21%

## Hardware Requirements
- GPU: NVIDIA GPU with 16GB+ VRAM
- Memory: 32GB+ RAM
- Storage: 150GB+ for dataset and checkpoints

## Training Tips
1. Use SWA for better generalization
2. Maintain consistent batch size of 256
3. Monitor GPU memory usage
4. Implement regular checkpointing
5. Use mixed-precision training for efficiency

## Future Improvements
1. Implement data augmentation strategies
2. Try different optimizers (SGD, AdamW)
3. Experiment with different learning rate schedules
4. Add model ensemble techniques
5. Implement more aggressive data preprocessing

## Repository Structure

```tex
.

├── train.py # Main training script

├── model.py # Model architecture

├── config.py # Training configuration

├── checkpoints/ # Model checkpoints

└── training.log # Training logs
```

# Additional Details

## Deployment Details
- **Instance Type**: AWS g4dn.2xlarge
- **GPU**: NVIDIA T4 (16GB VRAM)
- **vCPUs**: 8
- **Memory**: 32GB

## Training Timeline Analysis

### Initial Training Start
- Start Time: 2024-12-29 02:28:46
- Initial GPU Memory Usage: 1689MB/15360MB
- Initial Batch Processing Speed: 278.5 img/s

### Training Progress & Interruptions

1. **First Training Session**
   - Started: 2024-12-29 02:28:46
   - Initial Accuracy: 0.39%
   - GPU Utilization: 100%
   - Memory Usage: 11701MB/15360MB

2. **Major Training Progress**
   ```
   Epoch 1:  13.90% (2024-12-29 03:46:49)
   Epoch 5:  37.19% (2024-12-29 08:53:47)
   Epoch 6:  38.90% (2024-12-29 10:10:08)
   ```

3. **Final Session Before Interruption**
   - Last Recorded Time: 2025-01-01 00:44:40
   - Best Accuracy Achieved: 59.21%
   - Final GPU Memory State: 11781MB/15360MB

### Performance Metrics
- Average Training Speed: 282-291 images/second
- Time per Epoch: 1.26-1.30 hours
- Total Training Hours: ~72 hours (including interruptions)

### System Resource Utilization
- GPU Usage: Fluctuated between 52-100%
- GPU Memory: Consistently around 11.7GB/15.36GB
- CUDA Allocated Memory: Stable at 652MB

### Notable Interruptions
1. First Major Break:
   - 2024-12-31 17:54:50 to 2024-12-31 18:38:48

2. System Restart:
   - 2025-01-01 00:16:31
   - Multiple restart attempts between 00:18:21 and 00:28:12

3. Final Training State:
   - Last Recorded: 2025-01-01 17:05:03
   - Graceful shutdown with checkpoint saving

### Cost Efficiency
- Instance Type: g4dn.2xlarge
- Running Time: ~72 hours
- Effective Training Time: ~65 hours (excluding interruptions)
- Training Efficiency: 282.70 samples/second average

### Training Stability
- Initial GPU Memory: 1.8GB
- Stable GPU Memory: 11.7GB
- Consistent Processing Speed: ~285 img/s
- Regular Checkpoint Saves: Every 5 epochs

The training showed resilience to interruptions thanks to regular checkpointing, allowing for successful resumption after system breaks. The g4dn.2xlarge instance proved adequate for the task, maintaining stable performance throughout the training period.