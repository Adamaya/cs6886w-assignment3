import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


# ---------------------------------
# Reproducibility
# ---------------------------------


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make cuDNN deterministic (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Ensure hash-based ops are deterministic
    os.environ["PYTHONHASHSEED"] = str(seed)
    

# ---------------------------------
# Model definition
# ---------------------------------


def create_mobilenetv2_cifar10(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """
    Create a MobileNetV2 model adapted for CIFAR-10.
    """
    if pretrained:
        print("Loading Pretrained ImageNet weights for MobileNetV2")
        weights = MobileNet_V2_Weights.IMAGENET1K_V1
        model = mobilenet_v2(weights=weights)
    else:
        model = mobilenet_v2(weights=None)

    # Replace classifier (last linear layer) for CIFAR-10
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


# ---------------------------------
# Data
# ---------------------------------


def get_cifar10_loaders(
    data_dir: str,
    batch_size: int = 128,
    num_workers: int = 4,
):
    """
    Returns (trainloader, testloader) for CIFAR-10 with good augmentations.
    """
    # ImageNet-like normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # CIFAR-10 images are 32x32; we upscale to 224x224 for MobileNetV2
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.2),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )

    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return trainloader, testloader


# ---------------------------------
# Loss / metrics
# ---------------------------------


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy with label smoothing.
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, preds, target):
        num_classes = preds.size(1)
        log_preds = torch.log_softmax(preds, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_preds)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_preds, dim=1))


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    # Get top-k indices
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # Compare with targets expanded
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# ---------------------------------
# Training / evaluation
# ---------------------------------


def train_one_epoch(
    model: nn.Module,
    criterion,
    optimizer,
    dataloader,
    device,
    epoch: int,
    scaler=None,
):
    model.train()
    running_loss = 0.0
    running_top1 = 0.0
    total = 0

    start_time = time.time()

    for i, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        top1, = accuracy(outputs, targets, topk=(1,))
        bs = targets.size(0)
        running_loss += loss.item() * bs
        running_top1 += top1.item() * bs
        total += bs

        if (i + 1) % 50 == 0:
            avg_loss = running_loss / total
            avg_acc = running_top1 / total
            print(
                f"Epoch [{epoch}] Step [{i+1}/{len(dataloader)}] "
                f"Loss: {avg_loss:.4f} | Top-1: {avg_acc:.2f}%"
            )

    loss = running_loss / total
    acc1 = running_top1 / total
    elapsed = time.time() - start_time
    print(
        f"Epoch [{epoch}] TRAIN - Loss: {loss:.4f} | Top-1: {acc1:.2f}% "
        f"| Time: {elapsed:.1f}s"
    )
    return loss, acc1


@torch.no_grad()
def evaluate(model, criterion, dataloader, device, epoch: int):
    model.eval()
    running_loss = 0.0
    running_top1 = 0.0
    total = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        top1, = accuracy(outputs, targets, topk=(1,))
        bs = targets.size(0)
        running_loss += loss.item() * bs
        running_top1 += top1.item() * bs
        total += bs

    loss = running_loss / total
    acc1 = running_top1 / total
    print(f"Epoch [{epoch}] VALID - Loss: {loss:.4f} | Top-1: {acc1:.2f}%")
    return loss, acc1


# ---------------------------------
# Plotting
# ---------------------------------


def plot_curves(history, out_prefix: str = "mobilenetv2_cifar10"):
    """
    Plot train/val loss and accuracy curves and save them as PNGs.
    history: dict with keys:
        'train_loss', 'train_acc', 'val_loss', 'val_acc'
    """
    epochs = range(len(history["train_loss"]))

    # Loss curve
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val/Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss vs Epochs")
    plt.savefig(f"{out_prefix}_loss.png", bbox_inches="tight")
    plt.close()

    # Accuracy curve
    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train Top-1")
    plt.plot(epochs, history["val_acc"], label="Val/Test Top-1")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Accuracy vs Epochs")
    plt.savefig(f"{out_prefix}_accuracy.png", bbox_inches="tight")
    plt.close()


# ---------------------------------
# Argument parsing
# ---------------------------------


def get_args():
    parser = argparse.ArgumentParser(
        description="Train MobileNetV2 on CIFAR-10."
    )
    parser.add_argument("--data-dir", type=str, default="./data", help="Dataset root")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument(
        "--lr", type=float, default=0.05, help="Initial learning rate"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=4e-5, help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of dataloader workers"
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        help="Label smoothing factor",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable ImageNet pretraining for MobileNetV2",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./checkpoints/mobilenetv2_cifar10_best.pth",
        help="Path to save best checkpoint",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Use mixed precision training (AMP) when CUDA is available",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default="mobilenetv2_cifar10",
        help="Prefix for output loss/accuracy plots",
    )

    return parser.parse_args()


# ---------------------------------
# Main
# ---------------------------------


def main():
    args = get_args()
    print("Args:", vars(args))

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Data
    trainloader, testloader = get_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Model
    model = create_mobilenetv2_cifar10(
        num_classes=10,
        pretrained=not args.no_pretrained,
    )
    model = model.to(device)

    # Criterion
    criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing).to(device)

    # Optimizer with weight decay only on weights (not bias / norm)
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)

    optimizer = optim.SGD(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
        momentum=0.9,
        nesterov=True,
    )

    # Cosine LR schedule
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-4
    )

    # AMP scaler
    scaler = (
        torch.cuda.amp.GradScaler()
        if (args.mixed_precision and device == "cuda")
        else None
    )

    start_epoch = 0
    best_acc = 0.0
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    # Optionally resume from checkpoint
    if args.resume and os.path.isfile(args.resume):
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best_acc = checkpoint.get("best_acc", 0.0)
        start_epoch = checkpoint.get("epoch", 0) + 1
        print(
            f"Resumed from epoch {start_epoch}, best_acc={best_acc:.2f}%"
        )

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"\n=== Epoch {epoch}/{args.epochs - 1} ===")
        train_loss, train_acc = train_one_epoch(
            model, criterion, optimizer, trainloader, device, epoch, scaler
        )
        val_loss, val_acc = evaluate(
            model, criterion, testloader, device, epoch
        )

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_acc": best_acc,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }
            torch.save(state, args.save_path)
            print(
                f"New best acc: {best_acc:.2f}%. Saved checkpoint to {args.save_path}."
            )

    print(f"\nTraining finished. Best Top-1 accuracy: {best_acc:.2f}%")
    print(f"Best model saved to: {args.save_path}")

    # Plot curves
    plot_curves(history, out_prefix=args.out_prefix)
    print("Saved loss/accuracy curves to:")
    print(f" {args.out_prefix}_loss.png")
    print(f" {args.out_prefix}_accuracy.png")


if __name__ == "__main__":
    main()
