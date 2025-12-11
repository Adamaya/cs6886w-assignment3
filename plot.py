# plot_curves.py (optional helper)
import torch
import matplotlib.pyplot as plt

ckpt = torch.load("./checkpoints/mobilenetv2_cifar_best.pth", map_location="cpu")
train_losses = ckpt["train_losses"]
val_losses = ckpt["val_losses"]
train_accs = ckpt["train_accs"]
val_accs = ckpt["val_accs"]

epochs = range(1, len(train_losses) + 1)

plt.figure()
plt.plot(epochs, train_losses, label="train_loss")
plt.plot(epochs, val_losses, label="val_loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
plt.savefig("loss_curve.png")

plt.figure()
plt.plot(epochs, train_accs, label="train_acc")
plt.plot(epochs, val_accs, label="val_acc")
plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.legend()
plt.savefig("accuracy_curve.png")
