import torch.nn as nn
import torch.nn.functional as F
from resblock import ResBlock
from cbam import CBAM

class FCN(nn.Module):
    def __init__(self, out_dim=5):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),

            ResBlock(8),
            ResBlock(8),
            
            nn.Conv2d(8, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),

            ResBlock(16),
            ResBlock(16),

            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),

            ResBlock(32),

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

        self.cbam = CBAM(64)
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.dropout = nn.Dropout(p=0.2)
        self.regressor = nn.Linear(64, out_dim)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.cbam(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.regressor(x)

