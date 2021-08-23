import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg19 = models.vgg19(pretrained=True).features[:36].eval()
        self.mse = nn.MSELoss()

        for param in self.vgg19.parameters():
            param.requires_grad = False

    def forward(self, inputs, targets):
        inputs_features = self.vgg19(F.interpolate(inputs, (224, 224)))
        targets_features = self.vgg19(F.interpolate(targets, (224, 224)))
        return self.mse(inputs_features, targets_features)
