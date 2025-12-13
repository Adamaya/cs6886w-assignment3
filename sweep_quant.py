# sweep_quant.py
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb

from models.mobilenetv2_cifar import mobilenetv2_cifar
from utils.quantization import quantize_module_weights, ActivationQuantizer
from size_utils import (
    estimate_model_size_fp32,
    estimate_model_size_quantized,
)

def get_test_loader(batch_size=128):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    return DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            _, preds = out.max(1)
            correct += preds.eq(y).sum().item()
            total += y.size(0)
    return total_loss / total, 100.0 * correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/mobilenetv2_cifar_best.pth")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Wandb config – we’ll override per run
    wandb.init(project="cs6886-assign3", config={
        "weight_bits": 8,
        "act_bits": 8,
        "exclude_first_last": False,
    })
    cfg = wandb.config

    # Load baseline
    model = mobilenetv2_cifar(num_classes=10)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    # Baseline size
    baseline_mb = estimate_model_size_fp32(model)

    # Quantize weights
    model_q = quantize_module_weights(
        model, num_bits=cfg.weight_bits, exclude_first_last=cfg.exclude_first_last
    )

    # Simple activation quantization: final logits
    act_quant = ActivationQuantizer(num_bits=cfg.act_bits, signed=False)

    class QuantWrapper(nn.Module):
        def __init__(self, base, act_q):
            super().__init__()
            self.base = base
            self.act_q = act_q
        def forward(self, x):
            out = self.base(x)
            return self.act_q(out)

    model_q = QuantWrapper(model_q, act_quant).to(device)

    # Quantized size
    quant_mb, main_mb, overhead_mb = estimate_model_size_quantized(
        model_q, weight_bits=cfg.weight_bits
    )
    comp_ratio = baseline_mb / quant_mb

    test_loader = get_test_loader(batch_size=args.batch_size)
    loss, acc = evaluate(model_q, test_loader, device)

    wandb.log({
        "test_loss": loss,
        "test_acc": acc,
        "baseline_size_mb": baseline_mb,
        "quant_size_mb": quant_mb,
        "quant_main_mb": main_mb,
        "quant_overhead_mb": overhead_mb,
        "compression_ratio_model": comp_ratio,
        "weight_bits": cfg.weight_bits,
        "act_bits": cfg.act_bits,
        "exclude_first_last": cfg.exclude_first_last,
    })

if __name__ == "__main__":
    main()
