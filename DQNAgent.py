import numpy as np

import torch
import torch.nn as nn

class Neural_Network(nn.Module):
    def __init__(self, input_size=32, lr=0.002):
        super().__init__()
        """
        Pass for getting action
        Input anatomy:
        output anatomy:
            0. left
            1. right
            2. up
            3. down
        """
        self.model = nn.Sequential(
            nn.Linear(input_size, 20),
            nn.ReLU(),
            nn.Linear(20, 12),
            nn.ReLU(),
            nn.Linear(12, 4),
            nn.Sigmoid()
        )

        self.loss_f = nn.MSELoss()
        self.optim = torch.optim.Adam(self.parameters(), lr=lr)
    
    def forward(self, input_tensor):
        return self.model(input_tensor)

class QAgent:
    def __init__(self, net):
        self.net = net