# utils/quantization.py
import torch
import torch.nn as nn

def _get_qparams_minmax(x, num_bits, signed=True, eps=1e-8):
    x_min = x.min()
    x_max = x.max()

    if signed:
        qmin = -2 ** (num_bits - 1)
        qmax = 2 ** (num_bits - 1) - 1
    else:
        qmin = 0
        qmax = 2 ** num_bits - 1

    scale = (x_max - x_min) / float(qmax - qmin + eps)
    if scale.abs() < eps:
        scale = torch.tensor(1.0, device=x.device, dtype=x.dtype)

    zero_point = qmin - torch.round(x_min / scale)
    return scale, zero_point, qmin, qmax


def fake_quantize_tensor(x, num_bits=8, signed=True):
    """
    Fake quantization: quantize then dequantize.
    Returns x_hat, scale, zero_point.
    """
    with torch.no_grad():
        scale, zp, qmin, qmax = _get_qparams_minmax(x, num_bits, signed)
        q = torch.round(x / scale + zp)
        q.clamp_(qmin, qmax)
        x_hat = (q - zp) * scale
    return x_hat, scale, zp


def quantize_module_weights(module, num_bits=8, exclude_first_last=False):
    """
    Post-training quantization of Conv2d & Linear weights in-place.
    """
    modules = [m for m in module.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    n = len(modules)

    for idx, m in enumerate(modules):
        if exclude_first_last and (idx == 0 or idx == n - 1):
            continue

        with torch.no_grad():
            w = m.weight.data
            w_hat, scale, zp = fake_quantize_tensor(w, num_bits=num_bits, signed=True)
            m.weight.data.copy_(w_hat)
            # you can store scale/zp on the module if you want:
            m.register_buffer(f"weight_scale_{num_bits}b", scale)
            m.register_buffer(f"weight_zp_{num_bits}b", zp)

    return module


class ActivationQuantizer(nn.Module):
    """
    Module to quantize activations during forward pass.
    Per-tensor min/max per batch.
    """

    def __init__(self, num_bits=8, signed=False):
        super().__init__()
        self.num_bits = num_bits
        self.signed = signed

    def forward(self, x):
        x_hat, _, _ = fake_quantize_tensor(x, num_bits=self.num_bits, signed=self.signed)
        return x_hat
