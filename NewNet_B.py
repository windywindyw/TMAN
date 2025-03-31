import torch.nn as nn
from torchvision import models


class BreastBiomarkerNet(nn.Module):
    def __init__(self, num_classes=8):
        super(BreastBiomarkerNet, self).__init__()
        self.backbone = models.convnext_tiny(weights="DEFAULT")
        self.backbone.classifier[2] = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = self.backbone.classifier(x)

        return x
