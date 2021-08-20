import torch.nn as nn
from torchvision import models


class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg19 = models.vgg19(pretrained=True).features[:36].eval()
        self.loss = nn.MSELoss()

        for param in self.vgg19.parameters():
            param.requires_grad = False
	
    def forward(self, inputs, targets):
        inputs_features = self.vgg19(inputs)
        targets_features = self.vgg19(targets)
        return self.loss(inputs_features, targets_features)
