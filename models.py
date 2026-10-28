import torch
from torch import nn
from torchvision.models import vgg16, VGG16_Weights


def default_cnn(num_classes: int = 8, dropout: float = 0.5) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 224 -> 112
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 112 -> 56
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 56 -> 28
        nn.Flatten(),
        nn.Linear(128 * 28 * 28, 128), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(128, num_classes),
    )


class CustomCNN(nn.Module):
    def __init__(self, num_classes: int = 8, n_blocks: int = 4, base_filters: int = 32, kernel_size: int = 3,
                 dropout: float = 0.3, max_filters: int = 512, min_spatial: int = 7, input_size: int = 224) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_ch, size = 3, input_size
        for i in range(n_blocks):
            out_ch = min(base_filters * 2**i, max_filters)
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size, padding="same"), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size, padding="same"), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            ]
            if size > min_spatial:
                layers.append(nn.MaxPool2d(2))
                size //= 2
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(in_ch, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


_VGG_TRAINABLE = {"none": None, "last_layer": 28, "last_block": 24, "all": 0}


def vgg16_pretrained(num_classes: int = 8, trainable: str = "none", dropout: float = 0.5) -> nn.Module:

    if trainable not in _VGG_TRAINABLE:
        raise ValueError(f"trainable must be one of {list(_VGG_TRAINABLE)}, got {trainable!r}")
    m = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    for p in m.features.parameters():
        p.requires_grad = False
    start = _VGG_TRAINABLE[trainable]
    if start is not None:
        for p in m.features[start:].parameters():
            p.requires_grad = True
    m.classifier[2] = nn.Dropout(dropout)
    m.classifier[5] = nn.Dropout(dropout)
    m.classifier[6] = nn.Linear(4096, num_classes)
    return m


def build_model(name: str, **kwargs) -> nn.Module:
    registry = {"default": default_cnn, "custom": CustomCNN, "vgg16": vgg16_pretrained}
    return registry[name](**kwargs)
