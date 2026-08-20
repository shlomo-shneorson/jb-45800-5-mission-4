import torch.nn as nn
import torchvision.models as models


class WasteClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super(WasteClassifier, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)
