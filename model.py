import torch.nn as nn

class SnoutNet(nn.Module):
    def __init__(self):
        super(SnoutNet, self).__init__()
        # Define convolutional layers based on the architecture
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # Conv1: 3x227x227 -> 64x227x227
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Pool1: 64x227x227 -> 64x113x113
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Conv2: 64x113x113 -> 128x113x113
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Pool2: 128x113x113 -> 128x56x56
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # Conv3: 128x56x56 -> 256x56x56
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Pool3: 256x56x56 -> 256x28x28
        )

        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(256 * 28 * 28, 4096),  # FC1: 256x28x28 -> 4096
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1024),  # FC2: 4096 -> 1024
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2)  # FC3: 1024 -> 2 (final output, uv coordinates)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten before feeding into the fully connected layers
        x = self.classifier(x)
        return x
