# models/mobilenetv2_cifar.py
import torch
import torch.nn as nn

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(
                in_planes, out_planes, kernel_size,
                stride, padding, groups=groups, bias=False
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True),
        )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.append(
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim)
        )
        layers.append(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)
        )
        layers.append(nn.BatchNorm2d(oup))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)

class MobileNetV2_CIFAR(nn.Module):
    """
    MobileNetV2 adapted for CIFAR-10:
    - 32x32 inputs
    - no initial stride-2
    - num_classes=10
    """

    def __init__(self, num_classes=10, width_mult=1.0, round_nearest=8, dropout=0.2):
        super().__init__()

        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        # t, c, n, s
        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        def _make_divisible(v, divisor, min_value=None):
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        if width_mult != 1.0:
            input_channel = int(input_channel * width_mult)
            last_channel = int(last_channel * width_mult)

        input_channel = _make_divisible(input_channel, round_nearest)
        last_channel = _make_divisible(last_channel, round_nearest)

        features = []
        # First conv: stride 1 (CIFAR-10)
        features.append(ConvBNReLU(3, input_channel, kernel_size=3, stride=1))

        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    block(input_channel, output_channel, stride, expand_ratio=t)
                )
                input_channel = output_channel

        features.append(ConvBNReLU(input_channel, last_channel, kernel_size=1))

        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(last_channel, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

def mobilenetv2_cifar(num_classes=10, width_mult=1.0, dropout=0.2):
    return MobileNetV2_CIFAR(num_classes=num_classes, width_mult=width_mult, dropout=dropout)

if __name__ == "__main__":
    m = mobilenetv2_cifar()
    x = torch.randn(4, 3, 32, 32)
    print(m(x).shape)   # torch.Size([4, 10])
