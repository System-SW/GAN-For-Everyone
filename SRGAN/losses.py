import torch
import torchvision.models as models


class VGGPerceptualLoss(torch.nn.Module):
    """VGG LOSS for Perceptual loss
    Copy from https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49
    """

    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, inputs, target, feature_layers=[0, 1, 2, 3]):
        if inputs.shape[1] != 3:
            inputs = inputs.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        inputs = (inputs - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            inputs = self.transform(
                inputs, mode="bilinear", size=(224, 224), align_corners=False
            )
            target = self.transform(
                target, mode="bilinear", size=(224, 224), align_corners=False
            )
        loss = 0.0
        x = inputs
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
        return loss
