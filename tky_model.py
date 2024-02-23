import torch.nn as nn
import torch
import numpy as np
C0 = 0.28209479177387814

def SH2RGB(sh):
    return sh * C0 + 0.5
class PointCloud(nn.Module):
    def __init__(self, n_points, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # initialization taken from DreamGaussian
        phis = np.random.random((n_points,)) * 2 * np.pi
        costheta = np.random.random((n_points,)) * 2 - 1
        thetas = np.arccos(costheta)
        mu = np.random.random((n_points,))
        radius = 0.5 * np.cbrt(mu)
        x = radius * np.sin(thetas) * np.cos(phis)
        y = radius * np.sin(thetas) * np.sin(phis)
        z = radius * np.cos(thetas)
        xyz = np.stack((x, y, z), axis=1)

        shs = SH2RGB(np.random.random((n_points, 3)) / 255)

        self.points = nn.Parameter(torch.tensor(xyz, dtype=torch.float32).requires_grad_(True))
        self.colors = nn.Parameter(torch.tensor(shs, dtype=torch.float32).requires_grad_(True))
        self.opacity = nn.Parameter(torch.tensor(1.0, dtype=torch.float32).requires_grad_(True))
    
    def forward(self):
        return self.points.unsqueeze(0), self.colors.clip(0, 1).unsqueeze(0)