import torch.nn as nn
import torch

class FusionClassifier(nn.Module):
    def __init__(self, image_dim, text_dim, num_classes=5):
        super().__init__()
        self.fc1 = nn.Linear(image_dim + text_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, image_feat, text_feat):
        x = torch.cat((image_feat, text_feat), dim=1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
