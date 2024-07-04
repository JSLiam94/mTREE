import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Projection Head
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, H, W):
        super(ProjectionHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, H*W)
        self.H = H
        self.W = W

    def forward(self, x):
        x = F.relu(self.fc1(x.float()))
        x = self.fc2(x)
        x = x.view(-1, self.H, self.W)  # reshape to 2D
        # Optionally, use softmax if you want the output to be probabilities
        # x = F.softmax(x.view(-1, self.H*self.W), dim=-1).view(-1, self.H, self.W)
        return x