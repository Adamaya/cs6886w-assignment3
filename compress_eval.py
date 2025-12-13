# compress_eval.py
import torch
import torch.nn as nn
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2

def count_weight_params(model):
    total = 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            total += m.weight.numel()
    return total

def estimate_model_size_fp32(model):
    num_params = count_weight_params(model)
    bits = num_params * 32
    return bits / 8 / (1024 ** 2)   # MB

def estimate_model_size_quantized(model, weight_bits=8, overhead_per_tensor_bytes=8):
    """
    overhead_per_tensor_bytes ~ scale (4 bytes) + zero_point (4 bytes) per tensor.
    """
    num_params = 0
    num_tensors = 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            num_params += m.weight.numel()
            num_tensors += 1

    bits = num_params * weight_bits
    bytes_main = bits / 8
    bytes_overhead = num_tensors * overhead_per_tensor_bytes
    size_mb = (bytes_main + bytes_overhead) / (1024 ** 2)
    return size_mb, bytes_main / (1024 ** 2), bytes_overhead / (1024 ** 2)

def create_mobilenetv2_cifar10(num_classes: int = 10) -> nn.Module:
    """
    Create a MobileNetV2 model adapted for CIFAR-10 (matching train_mobilenetv2.py).
    """
    model = mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

def get_test_loader(batch_size=128):
    """Match the test transform from train_mobilenetv2.py"""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)
    return DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

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
    Measure activations for a single forward pass using the same transforms.
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

    # Use same transforms as test
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
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
    bytes_fp32 = total_elements * 4
    bytes_q = total_elements * (act_bits / 8.0)

    return bytes_fp32, bytes_q

def quantize_module_weights(model, num_bits=8, exclude_first_last=False):
    """Placeholder for quantization logic - implement based on your needs"""
    # Add your quantization implementation here
    return model

class ActivationQuantizer(nn.Module):
    """Placeholder for activation quantization - implement based on your needs"""
    def __init__(self, num_bits=8, signed=False):
        super().__init__()
        self.num_bits = num_bits
        self.signed = signed
    
    def forward(self, x):
        # Add your activation quantization logic here
        return x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/mobilenetv2_cifar10_best.pth")
    parser.add_argument("--weight_bits", type=int, default=8)
    parser.add_argument("--act_bits", type=int, default=8)
    parser.add_argument("--exclude_first_last", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model (matching train_mobilenetv2.py structure)
    model = create_mobilenetv2_cifar10(num_classes=10)
    ckpt = torch.load(args.checkpoint, map_location=device)

    # Handle checkpoint format from train_mobilenetv2.py
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        print(f"Loaded checkpoint from epoch {ckpt.get('epoch', 'unknown')}")
        print(f"Best accuracy from training: {ckpt.get('best_acc', 'unknown'):.2f}%")
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.to(device)
    
    # Baseline sizes (weights only)
    weight_params = count_weight_params(model)
    baseline_model_mb = estimate_model_size_fp32(model)

    print(f"\nBaseline model size (weights only, FP32): {baseline_model_mb:.4f} MB")
    print(f"Total weight parameters: {weight_params:,}")

    # Quantize weights
    model_q = quantize_module_weights(
        model, num_bits=args.weight_bits, exclude_first_last=args.exclude_first_last
    )

    # Activation quantization wrapper
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

    # Quantized weight size
    quant_model_mb, main_mb, overhead_mb = estimate_model_size_quantized(
        model_q, weight_bits=args.weight_bits
    )

    print(f"\nQuantized model size (weights + overhead): {quant_model_mb:.4f} MB")
    print(f"  main weight data: {main_mb:.4f} MB")
    print(f"  overhead (scales/zps): {overhead_mb:.6f} MB")
    print(f"Compression ratio (weights): {baseline_model_mb / quant_model_mb:.2f}x")

    # Activation sizes
    bytes_fp32, bytes_q = measure_activation_bytes(model_q, device, act_bits=args.act_bits)
    act_ratio = bytes_fp32 / bytes_q if bytes_q > 0 else 1.0

    print(f"\nActivation bytes (FP32): {bytes_fp32 / (1024**2):.4f} MB")
    print(f"Activation bytes (quantized, {args.act_bits} bits): {bytes_q / (1024**2):.4f} MB")
    print(f"Compression ratio (activations): {act_ratio:.2f}x")

    # Evaluate accuracy
    print("\nEvaluating model...")
    test_loader = get_test_loader(batch_size=args.batch_size)
    loss, acc = evaluate(model_q, test_loader, device)
    print(f"Quantized Test Loss: {loss:.4f}, Test Acc: {acc:.2f}%")

if __name__ == "__main__":
    main()