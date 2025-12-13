# compress_eval.py
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.mobilenetv2_cifar import mobilenetv2_cifar
from utils.quantization import quantize_module_weights, ActivationQuantizer


# -------- size utilities -------- #

def count_weight_params(model):
    total = 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            total += m.weight.numel()
    return total


def estimate_model_size_fp32(model):
    """
    We treat all Conv/Linear weights as FP32 (4 bytes each).
    """
    num_params = count_weight_params(model)
    bits = num_params * 32
    return bits / 8 / (1024 ** 2)   # MB


def estimate_model_size_quantized(model, weight_bits=8, overhead_per_tensor_bytes=8):
    """
    Estimate model size when Conv/Linear weights are stored with 'weight_bits' bits.
    We add overhead_per_tensor_bytes for each quantized tensor to store scale+zero_point.

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


# -------- data + eval utilities -------- #

def get_test_loader(batch_size=128):
    # IMPORTANT: use CIFAR-10 stats, same as training
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_set = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    return DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )


def evaluate(model, loader, device):
    criterion = nn.CrossEntropyLoss()
    model.eval()
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


def measure_activation_bytes(model, device, act_bits=8, batch_size=32):
    """
    Measure activation size for ONE forward pass on the test set.
    We hook conv+ReLU6+Linear modules, count elements, and compute bytes for FP32 vs act_bits.
    """
    hooks = []
    act_sizes = []

    def hook_fn(module, inp, out):
        if isinstance(out, torch.Tensor):
            act_sizes.append(out.numel())
        elif isinstance(out, (list, tuple)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
            act_sizes.append(out[0].numel())

    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.ReLU6, nn.Linear)):
            hooks.append(m.register_forward_hook(hook_fn))

    # one dummy batch from test set
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

    x, _ = next(iter(dl))
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/mobilenetv2_cifar_best.pth")
    parser.add_argument("--weight_bits", type=int, default=8)
    parser.add_argument("--act_bits", type=int, default=8)
    parser.add_argument("--exclude_first_last", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"==> Using weight_bits={args.weight_bits}, act_bits={args.act_bits}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load baseline model
    model = mobilenetv2_cifar(num_classes=10)
    ckpt = torch.load(args.checkpoint, map_location=device)

    # robust checkpoint loading
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    # fix classifier name mismatch if needed
    if "classifier.1.weight" in state_dict and "classifier.weight" not in state_dict:
        state_dict["classifier.weight"] = state_dict.pop("classifier.1.weight")
        state_dict["classifier.bias"] = state_dict.pop("classifier.1.bias")
        for k in list(state_dict.keys()):
            if k.startswith("classifier.0."):
                state_dict.pop(k)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("Missing keys when loading:", missing)
    print("Unexpected keys when loading:", unexpected)

    model.to(device)

    # Baseline size (weights, FP32)
    baseline_model_mb = estimate_model_size_fp32(model)
    print(f"Baseline model size (weights only, FP32): {baseline_model_mb:.4f} MB")

    # Quantize weights (post-training) to the specified bit-width
    model_q = quantize_module_weights(
        model, num_bits=args.weight_bits, exclude_first_last=args.exclude_first_last
    )

    # Wrap model with activation quantizer on outputs (for simplicity)
    act_q = ActivationQuantizer(num_bits=args.act_bits, signed=False)

    class Wrapped(nn.Module):
        def __init__(self, base, aq):
            super().__init__()
            self.base = base
            self.aq = aq
        def forward(self, x):
            out = self.base(x)
            return self.aq(out)

    model_q = Wrapped(model_q, act_q).to(device)

    # Quantized weight size (estimate)
    quant_model_mb, main_mb, overhead_mb = estimate_model_size_quantized(
        model_q, weight_bits=args.weight_bits
    )
    print(f"Quantized model size (weights + overhead): {quant_model_mb:.4f} MB")
    print(f"  main weight data: {main_mb:.4f} MB")
    print(f"  overhead (scales/zps): {overhead_mb:.6f} MB")

    if quant_model_mb > 0:
        print(f"Compression ratio (weights): {baseline_model_mb / quant_model_mb:.2f}x")

    # Activation sizes
    bytes_fp32, bytes_q = measure_activation_bytes(
        model_q, device, act_bits=args.act_bits, batch_size=32
    )
    act_ratio = bytes_fp32 / bytes_q if bytes_q > 0 else 1.0

    print(f"\nActivation bytes (FP32): {bytes_fp32 / (1024**2):.4f} MB")
    print(f"Activation bytes (quantized, {args.act_bits} bits): {bytes_q / (1024**2):.4f} MB")
    print(f"Compression ratio (activations): {act_ratio:.2f}x\n")

    # Evaluate accuracy
    print("Evaluating model...")
    test_loader = get_test_loader(batch_size=args.batch_size)
    loss, acc = evaluate(model_q, test_loader, device)
    print(f"Quantized Test Loss: {loss:.4f}, Test Acc: {acc:.2f}%")

if __name__ == "__main__":
    main()
