# model.py
import torch.nn as nn
from torchvision import models

def get_resnet18_model(num_classes, pretrained=True):
    """
    Returns a resnet18 adapted for multi-label classification.
    Final activation is Sigmoid and loss used in training will be BCELoss.
    """
    model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
    # freeze backbone by default (uncomment if you want to finetune)
    for param in model.parameters():
        param.requires_grad = False

    # replace head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
        nn.Sigmoid()
    )
    return model
