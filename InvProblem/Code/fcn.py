import torch
import torch.nn as nn
import torch.nn.functional as F
from resblock import ResBlock
from cbam import CBAM
from cbam import ChannelAttention

class FCN(nn.Module):
    def __init__(self, out_dim=5):
        super().__init__()

        #stage1: 8x8x8
        self.stage1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.01, inplace=True),
            ResBlock(8),
            ResBlock(8)
        )

        #stage2: 16x8x8
        self.stage2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.01, inplace=True),
            ResBlock(16),
            ResBlock(16)
        )
        self.cbam = CBAM(16)
        
        #stage3: 32x4x4
        self.stage3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride = 2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01, inplace=True),
            ResBlock(32),
            ResBlock(32)
        )

        #stage4: 64x4x4
        self.stage4 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01, inplace=True),
            ResBlock(64)
        )
        self.ca = ChannelAttention(64)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.shared = nn.Sequential(
            nn.Linear(64, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01, inplace=True),
        )

        self.head_xyz = nn.Linear(64, 3)

        self.head_ang = nn.Sequential(
            nn.Linear(64, 16),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(16, 2),
            nn.Tanh()           
        )

    def forward(self, x):
        # Feature extraction
        x = self.stage1(x)      
        x = self.stage2(x) 
        x = self.cbam(x)     
        x = self.stage3(x)      
        x = self.stage4(x)      

        # Attention + pooling
        x = self.ca(x)        
        x = self.pool(x)       
        x = x.flatten(1)

        # Shared MLP
        feat = self.shared(x)   

        # Dual head output
        xyz = self.head_xyz(feat)           
        ang = self.head_ang(feat)           

        return torch.cat([xyz, ang], dim=1) # B x 5