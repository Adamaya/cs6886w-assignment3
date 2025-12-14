# cs6886w-assignment3

This repository contains the implementation for Assignment 3 of CS6886W - Systems for Deep Learning.

## Environment Setup

### Prerequisites
- Download and install Anaconda from the [official website](https://www.anaconda.com/download)

### Installation Steps
Follow these commands to configure the runtime environment:

```bash
conda create -n vgg6 python=3.10 -y
conda activate vgg6
pip install torch torchvision matplotlib pandas scikit-learn tqdm seaborn wandb
```

## Usage

### Question 1: Training Baseline Model

Train the MobileNetV2 model on CIFAR-10 dataset:

```bash
python train_mobilenetv2.py \
    --data-dir ./data \
    --epochs 50 \
    --batch-size 128 \
    --lr 0.05 \
    --weight-decay 4e-5 \
    --label-smoothing 0.1 \
    --mixed-precision
```

**Training Parameters:**
- Epochs: 50
- Batch Size: 128
- Learning Rate: 0.05
- Weight Decay: 4e-5
- Label Smoothing: 0.1
- Mixed Precision Training: Enabled

### Model Compression and Evaluation

Evaluate the compressed model with quantization:

```bash
python compress_eval.py \
    --checkpoint ./checkpoints/mobilenetv2_cifar10_best.pth \
    --weight_bits 8 \
    --act_bits 8 \
    --batch_size 128 \
    --device cuda
```

**Quantization Settings:**
- Weight Quantization: 8-bit
- Activation Quantization: 8-bit
- Device: CUDA (GPU)

## Project Structure

```
cs6886w-assignment3/
├── train_mobilenetv2.py    # Training script
├── compress_eval.py         # Compression evaluation script
├── checkpoints/             # Model checkpoints directory
└── data/                    # Dataset directory
```

## Requirements
- Python 3.10
- PyTorch
- torchvision
- CUDA-enabled GPU (recommended)
