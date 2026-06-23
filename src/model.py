# src/model.py
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import timm
import torch.nn as nn

def create_resnet34(num_classes=43, pretrained=True):
    model = timm.create_model(
        'resnet34',
        pretrained=pretrained,
        num_classes=num_classes
    )
    return model