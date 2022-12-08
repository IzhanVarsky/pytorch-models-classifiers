import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from utils import EarlyStopping, train_model


def get_resnet_model(num_classes, device, pretrained):
    model = models.resnet50(pretrained=pretrained).to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(inplace=True),
        nn.Linear(512, num_classes)).to(device)
    return model


def train_resnet(dataloaders, image_datasets, num_classes, device, num_epochs=50):
    model = get_resnet_model(num_classes, device, pretrained=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters())
    early_stopping = EarlyStopping(model_name="ResNetTransfer_check", save_best=True)
    torch.cuda.empty_cache()
    return train_model(model, dataloaders, image_datasets, criterion,
                       optimizer, device, early_stopping, num_epochs)


def get_resnet_classifier(num_classes, device):
    from classifier import Classifier
    model = get_resnet_model(num_classes, device, pretrained=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters())
    return Classifier(model, optimizer, criterion, device, "ResNetTransfer_pretrained")
