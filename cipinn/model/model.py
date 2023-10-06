import torch
from torch import nn
import numpy as np


class FcNet(nn.Module):
    def __init__(self, db=20, depth=8, dx=2, dy=1):
        super(FcNet, self).__init__()
        self.depth = depth
        self.db = db
        fc = []
        for i in range(depth + 1):
            if i == 0:
                fc.append(nn.Linear(dx, db))
            elif i == depth:
                fc.append(nn.Linear(db, dy))
            else:
                fc.append(nn.Linear(db, db))
        self.fc = nn.ModuleList(fc)
        self.randominit()

    def forward(self, x):
        for i in range(self.depth):
            x = torch.tanh(self.fc[i](x))
        return self.fc[self.depth](x)

    def randominit(self):
        for i in range(self.depth + 1):
            out_dim, in_dim = self.fc[i].weight.shape
            xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
            self.fc[i].weight.data.normal_(0, xavier_stddev)
            self.fc[i].bias.data.fill_(0.0)
