import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_mobilenetv2 import (
    create_mobilenetv2_cifar10,
    get_cifar10_loaders,
    set_seed,
)

from utils.quantization import quantize_module_weights, ActivationQuantizer


# -------- size utilities -------- #

def count_weight_params(model: nn.Module) -> int:
    total = 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            total += m.weight.numel()
    return total


def estimate_model_size_fp32(model: nn.Module) -> float:
    """
    Treat all Conv/Linear weights as FP32 (4 bytes each).
    """
    num_params = count_weight_params(model)
    bits = num_params * 32
    return bits / 8 / (1024 ** 2)   # MB


def estimate_model_size_quantized(
    model: nn.Module,
    weight_bits: int = 8,
    overhead_per_tensor_bytes: int = 8,
):
    """
    Estimate model size when Conv/Linear weights are stored with 'weight_bits' bits.
    We add overhead_per_tensor_bytes for each quantized tensor to store scale, etc.

    NOTE: This is purely an estimate; actual PyTorch tensors are still FP32 in memory.
    """
    num_params = 0
    num_tensors = 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            num_params += m.weight.numel()
            num_tensors += 1

    bits = num_params * weight_bits
    bytes_main = bits / 8.0
    bytes_overhead = num_tensors * float(overhead_per_tensor_bytes)

    size_mb = (bytes_main + bytes_overhead) / (1024.0 ** 2)
    main_mb = bytes_main / (1024.0 ** 2)
    overhead_mb = bytes_overhead / (1024.0 ** 2)

    return size_mb, main_mb, overhead_mb


# -------- eval utilities -------- #

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        total_loss += loss.item() * x.size(0)
        _, preds = out.max(1)
        correct += preds.eq(y).sum().item()
        total += y.size(0)
    return total_loss / total, 100.0 * correct / total


def measure_activation_bytes(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    act_bits: int = 8,
    batch_size: int = 32,
):
    """
    Measure activation size for ONE forward pass on the test set.

    We hook Conv2d / ReLU6 / Linear modules, count elements, and compute bytes
    for FP32 vs act_bits.
    """
    hooks = []
    act_sizes = []

    def hook_fn(module, inp, out):
        if isinstance(out, torch.Tensor):
            act_sizes.append(out.numel())
        elif (
            isinstance(out, (list, tuple))
            and len(out) > 0
            and isinstance(out[0], torch.Tensor)
        ):
            act_sizes.append(out[0].numel())

    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.ReLU6, nn.Linear)):
            hooks.append(m.register_forward_hook(hook_fn))

    # one batch from given loader
    it = iter(loader)
    x, _ = next(it)
    if batch_size is not None and x.size(0) > batch_size:
        x = x[:batch_size]
    x = x.to(device)

    with torch.no_grad():
        _ = model(x)

    for h in hooks:
        h.remove()

    total_elements = sum(act_sizes)
    bytes_fp32 = total_elements * 4.0                # 32 bits
    bytes_q = total_elements * (act_bits / 8.0)      # act_bits

    return bytes_fp32, bytes_q


# -------- main -------- #

def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate quantized MobileNetV2 on CIFAR-10."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Dataset root",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/mobilenetv2_cifar10_best.pth",
        help="Path to trained MobileNetV2 checkpoint",
    )
    parser.add_argument(
        "--weight_bits",
        type=int,
        default=8,
        help="Bit-width for weight quantization",
    )
    parser.add_argument(
        "--act_bits",
        type=int,
        default=8,
        help="Bit-width for activation size estimation",
    )
    parser.add_argument(
        "--exclude_first_last",
        action="store_true",
        help="Exclude first and last Conv/Linear from quantization",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()

def attach_activation_quantization(model: nn.Module, act_bits: int = 8, signed: bool = False):
    """
    Attach fake activation quantization after ReLU / ReLU6 layers using forward hooks.
    This does *not* change the architecture or checkpoints.
    """
    quantizers = []

    def make_hook(qmod):
        def hook(module, inp, out):
            # out is a tensor (or tuple); ActivationQuantizer will quantize it
            if isinstance(out, torch.Tensor):
                return qmod(out)
            elif isinstance(out, (list, tuple)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
                return type(out)([qmod(out[0])] + list(out[1:]))
            else:
                return out
        return hook

    for m in model.modules():
        # MobileNetV2 uses ReLU6; include ReLU just in case.
        if isinstance(m, (nn.ReLU6, nn.ReLU)):
            q = ActivationQuantizer(num_bits=act_bits, signed=signed)
            quantizers.append(q)
            m.register_forward_hook(make_hook(q))

    # Just to keep them alive (not strictly necessary if you don't need to access them later)
    model._activation_quantizers = quantizers
    return model


def main():
    args = get_args()
    print("Args:", vars(args))

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data (same transforms as in train_mobilenetv2.py)
    train_loader, test_loader = get_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Model definition aligned with training script
    model = create_mobilenetv2_cifar10(
        num_classes=10,
        pretrained=False,   # we will load our trained weights
    )

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("Missing keys when loading:", missing)
    print("Unexpected keys when loading:", unexpected)

    model.to(device)

    # Baseline FP32 size (weights only)
    baseline_model_mb = estimate_model_size_fp32(model)
    print(f"Baseline model size (weights only, FP32): {baseline_model_mb:.4f} MB")

    # Quantize weights (post-training)
    print(f"\n==> Quantizing weights to {args.weight_bits} bits...")
    model_q = quantize_module_weights(
        model,
        num_bits=args.weight_bits,
        exclude_first_last=args.exclude_first_last,
    ).to(device)

    # Attach activation fake-quantization
    print(f"==> Attaching activation quantization ({args.act_bits} bits)...")
    model_q = attach_activation_quantization(
        model_q,
        act_bits=args.act_bits,
        signed=True  # ReLU6 outputs are non-negative
    )

    # Quantized model size (estimate)
    quant_model_mb, main_mb, overhead_mb = estimate_model_size_quantized(
        model_q,
        weight_bits=args.weight_bits,
    )
    print(f"\nQuantized model size (weights + overhead): {quant_model_mb:.4f} MB")
    print(f"  main weight data: {main_mb:.4f} MB")
    print(f"  overhead (scales etc.): {overhead_mb:.6f} MB")
    if quant_model_mb > 0:
        print(f"Compression ratio (weights): {baseline_model_mb / quant_model_mb:.2f}x")

    # Activation sizes
    print("\nMeasuring activation memory for one forward pass...")
    bytes_fp32, bytes_q = measure_activation_bytes(
        model_q,
        test_loader,
        device,
        act_bits=args.act_bits,
        batch_size=32,
    )
    act_ratio = bytes_fp32 / bytes_q if bytes_q > 0 else 1.0
    print(f"Activation bytes (FP32): {bytes_fp32 / (1024**2):.4f} MB")
    print(f"Activation bytes (quantized, {args.act_bits} bits): {bytes_q / (1024**2):.4f} MB")
    print(f"Compression ratio (activations): {act_ratio:.2f}x\n")

    # Evaluate accuracy of quantized model
    print("Evaluating quantized model on CIFAR-10 test set...")
    loss, acc = evaluate(model_q, test_loader, device)
    print(f"Quantized Test Loss: {loss:.4f}, Test Acc: {acc:.2f}%")


if __name__ == "__main__":
    main()
