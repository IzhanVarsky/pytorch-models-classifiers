import torch
import torch.nn as nn
import torch.optim as optim

from utils import EarlyStopping, train_model


class VGG11(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super(VGG11, self).__init__()
        self.conv_layers = nn.Sequential(
            # 224x224x3
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            # 224x224x64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 112x112x64
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            # 112x112x128
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 56x56x128
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            # 56x56x256
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=512 * 7 * 7, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.flatten()
        x = self.linear_layers(x)
        return x


def get_vgg_model(num_classes, device):
    return VGG11(num_classes).to(device)


def train_vgg(dataloaders, image_datasets, num_classes, device, num_epochs=50):
    model = get_vgg_model(num_classes, device)

    criterion = nn.CrossEntropyLoss()
    lr = 0.0001
    optimizer = optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(model_name=f"VGG11_lr{lr}", patience=50, save_best=True)

    torch.cuda.empty_cache()
    return train_model(model, dataloaders, image_datasets, criterion,
                       optimizer, device, early_stopping, num_epochs)


def get_vgg11_classifier(num_classes, device):
    from classifier import Classifier
    model = get_vgg_model(num_classes, device)
    criterion = nn.CrossEntropyLoss()
    lr = 0.0001
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return Classifier(model, optimizer, criterion, device, "VGG11")
