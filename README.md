# cs6886w-assignment3

## Environment Setup to train and test the model
- Download the anaconda software and install it. [Download Link](https://www.anaconda.com/download)
- Run the following commands to configure the runtime environment:
```
conda create -n vgg6 python=3.10 -y
conda activate vgg6
pip install torch torchvision matplotlib pandas scikit-learn tqdm seaborn wandb
```

## Question 1. Training Baseline
```
python train_mobilenetv2.py --data-dir ./data --epochs 50 --batch-size 128 --lr 0.05  --weight-decay 4e-5 --label-smoothing 0.1 --mixed-precision

```
# Best Compression Evaluation
```
python compress_eval.py --checkpoint ./checkpoints/mobilenetv2_cifar10_best.pth --weight_bits 8 --act_bits 8 --batch_size 128 --device cuda
```
