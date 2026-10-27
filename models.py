import torch
from torch import nn
from torchvision.models import vgg16, VGG16_Weights

def default_cnn(num_classes: int = 8) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   #224 -> 112
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   #112 -> 56 
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  #56 -> 28 
        nn.Flatten(),
        nn.Linear(128 * 28 * 28, 128), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(128, num_classes),
    )

class CustomCNN(nn.Module):
    def __init__(self, num_classes: int = 8, n_blocks: int = 4, base_filters: int = 32, kernel_size: int = 3, dropout: float = 0.3) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = 3
        for i in range(n_blocks):
            out_ch = base_filters * 2**i
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size, padding=kernel_size // 2), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(in_ch, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))

def vgg16_pretrained(num_classes: int = 8, freeze: bool=True, unfreeze_last_block: bool = False) -> nn.Module:
    m = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    if freeze:
        for p in m.features.parameters():
            p.requires_grad = False
    if unfreeze_last_block:
        for p in m.features[24:].parameters():
            p.requires_grad = True
    m.classifier[6] = nn.Linear(4096, num_classes)
    return m

def build_model(name: str, **kwargs) -> nn.Module:
    registry = {"default": default_cnn, "custom": CustomCNN, "vgg16": vgg16_pretrained}
    return registry[name](**kwargs)
