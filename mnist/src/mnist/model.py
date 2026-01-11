import torch
from torch import nn

class MyAwesomeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Adding BatchNorm2d after Conv layers for 0-mean/1-std data
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.bn1 = nn.BatchNorm2d(32)  
        
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn1(self.conv1(x))) # BN before activation
        x = torch.max_pool2d(x, 2, 2)
        
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.max_pool2d(x, 2, 2)
        
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.max_pool2d(x, 2, 2)
        
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc1(x)