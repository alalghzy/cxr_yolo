import torch
import torch.nn as nn

class ECLA(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        reduced = max(1, c // 16)

        self.shared = nn.Sequential(
            nn.Conv2d(c, reduced, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, c, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.shared(self.avg_pool(x))
        max = self.shared(self.max_pool(x))
        out = self.sigmoid(avg + max)
        return x * out

