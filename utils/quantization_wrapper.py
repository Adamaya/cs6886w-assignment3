# utils/quantized_wrapper.py (optional file)
import torch.nn as nn
from utils.quantization import ActivationQuantizer

class QuantizedMobileNetWrapper(nn.Module):
    def __init__(self, base_model, act_bits=8):
        super().__init__()
        self.base = base_model
        self.act_q = ActivationQuantizer(num_bits=act_bits, signed=False)

    def forward(self, x):
        # quantize input activations
        x = self.act_q(x)
        x = self.base.features(x)
        x = self.base.avgpool(x)
        x = self.base.dropout(x)
        x = x.view(x.size(0), -1)
        # we could quantize here too if we want:
        x = self.act_q(x)
        x = self.base.classifier(x)
        return x
