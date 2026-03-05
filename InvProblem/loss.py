import torch
import torch.nn as nn
import torch.nn.functional as F

class HomoscedasticPoseLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_sigma_xyz = nn.Parameter(torch.zeros(1))
        self.log_sigma_ang = nn.Parameter(torch.zeros(1))

    def forward(self, pred, target):
        loss_xyz = F.mse_loss(
            pred[:, :3],
            target[:, :3],
            reduction="mean"
        )

        loss_ang = F.mse_loss(
            pred[:, 3:],
            target[:, 3:],
            reduction="mean"
        )
        
        loss = (
            torch.exp(-self.log_sigma_xyz) * loss_xyz
            + self.log_sigma_xyz
            + torch.exp(-self.log_sigma_ang) * loss_ang
            + self.log_sigma_ang
        )

        return loss
