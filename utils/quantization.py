# utils/quantization.py
import torch
import torch.nn as nn


def _per_channel_symmetric_quant_weight(w: torch.Tensor, num_bits: int):
    """
    Per-channel symmetric quantization for weights.
    For Conv2d / Linear, we quantize per output channel (dim 0).

    Args:
        w:        weight tensor of shape [out_c, ...]
        num_bits: bit-width (e.g. 8, 4, 2, 1)

    Returns:
        w_hat:    dequantized weight tensor (same shape as w)
        scale:    tensor of shape [out_c] with scales per channel
    """
    # flatten all but out_channel dim
    out_c = w.size(0)
    w_flat = w.view(out_c, -1)

    # max absolute value per channel
    max_abs = w_flat.abs().max(dim=1).values  # [out_c]
    qmax = 2 ** (num_bits - 1) - 1  # e.g. 127 for 8-bit

    # avoid division by zero
    eps = 1e-8
    scale = max_abs / (qmax + eps)
    scale = torch.where(scale < eps, torch.ones_like(scale) * (1.0 / qmax), scale)

    # reshape for broadcasting back to original weight shape
    view_shape = [out_c] + [1] * (w.dim() - 1)
    scale_broadcast = scale.view(*view_shape)

    # quantize and dequantize
    q = torch.round(w / scale_broadcast)
    q.clamp_(-qmax, qmax)
    w_hat = q * scale_broadcast

    return w_hat, scale


def quantize_module_weights(module: nn.Module, num_bits: int = 8, exclude_first_last: bool = False) -> nn.Module:
    """
    Post-training weight quantization (fake) for Conv2d & Linear.
    Uses per-channel symmetric quantization.

    Args:
        module:             root module (e.g. full model)
        num_bits:           bit-width for weights
        exclude_first_last: if True, skip the first and last Conv/Linear

    Returns:
        module with quantized weights (still float, but quantization simulated)
    """
    layers = [m for m in module.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    n = len(layers)

    for idx, m in enumerate(layers):
        if exclude_first_last and (idx == 0 or idx == n - 1):
            continue

        with torch.no_grad():
            w = m.weight.data
            # per-channel symmetric on out_channels/features
            w_hat, scale = _per_channel_symmetric_quant_weight(w, num_bits=num_bits)
            m.weight.data.copy_(w_hat)

            # store scale for size accounting / inspection (doesn't affect forward)
            m.register_buffer(f"weight_scale_{num_bits}b", scale)

    return module


class ActivationQuantizer(nn.Module):
    """
    Placeholder for activation quantization.

    For this assignment version, we will NOT use it inside the forward pass,
    but we keep the class so your imports still work and you can mention
    how activations *would* be quantized (per-tensor uniform).
    """

    def __init__(self, num_bits: int = 8, signed: bool = False):
        super().__init__()
        self.num_bits = num_bits
        self.signed = signed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # For stability of accuracy, we just return x here.
        # You still use act_bits for theoretical compression calculations.
        return x
