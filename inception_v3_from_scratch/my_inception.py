import torch
import torch.nn as nn
from torch import optim

from utils import EarlyStopping, train_model


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super(ConvBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, bias=False, **kwargs),
            nn.BatchNorm2d(out_channels, eps=0.001),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)


class GridReduction(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GridReduction, self).__init__()
        out0 = out_channels[0]
        out1 = out_channels[1]
        self.branch1 = nn.Sequential(
            ConvBlock(in_channels, out0, kernel_size=1),
            ConvBlock(out0, out0, kernel_size=3, padding=1),
            ConvBlock(out0, out0, kernel_size=3, stride=2)
        )
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, out1, kernel_size=1),
            ConvBlock(out1, out1, kernel_size=3, stride=2),
        )
        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, input_img):
        o1 = self.branch1(input_img)
        o2 = self.branch2(input_img)
        o3 = self.branch3(input_img)
        return torch.cat([o1, o2, o3], dim=1)


class InceptionX3(nn.Module):
    def __init__(self, in_channels, inner):
        super(InceptionX3, self).__init__()
        out_channels = in_channels // 4
        self.branch1 = nn.Sequential(
            ConvBlock(in_channels, inner, kernel_size=1),
            ConvBlock(inner, inner, kernel_size=3, padding=1),
            ConvBlock(inner, out_channels, kernel_size=3, padding=1)
        )
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, inner, kernel_size=1),
            ConvBlock(inner, out_channels, kernel_size=3, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, out_channels, kernel_size=1)
        )
        self.branch4 = ConvBlock(in_channels, out_channels, kernel_size=1)

    def forward(self, input_img):
        o1 = self.branch1(input_img)
        o2 = self.branch2(input_img)
        o3 = self.branch3(input_img)
        o4 = self.branch4(input_img)
        return torch.cat([o1, o2, o3, o4], dim=1)


class InceptionX5(nn.Module):
    def __init__(self, in_channels, inner, n=7):
        super(InceptionX5, self).__init__()
        n2 = n // 2
        out_channels = in_channels // 4

        self.branch1 = nn.Sequential(
            ConvBlock(in_channels, inner, kernel_size=1),
            ConvBlock(inner, inner, kernel_size=(1, n), padding=(0, n2)),
            ConvBlock(inner, inner, kernel_size=(n, 1), padding=(n2, 0)),
            ConvBlock(inner, inner, kernel_size=(1, n), padding=(0, n2)),
            ConvBlock(inner, out_channels, kernel_size=(n, 1), padding=(n2, 0)),
        )
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, inner, kernel_size=1),
            ConvBlock(inner, inner, kernel_size=(1, n), padding=(0, n2)),
            ConvBlock(inner, out_channels, kernel_size=(n, 1), padding=(n2, 0)),
        )
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, out_channels, kernel_size=1)
        )
        self.branch4 = ConvBlock(in_channels, out_channels, kernel_size=1)

    def forward(self, input_img):
        o1 = self.branch1(input_img)
        o2 = self.branch2(input_img)
        o3 = self.branch3(input_img)
        o4 = self.branch4(input_img)
        return torch.cat([o1, o2, o3, o4], dim=1)


class InceptionX2(nn.Module):
    def __init__(self, in_fts, out_fts):
        super(InceptionX2, self).__init__()
        self.branch1 = nn.Sequential(
            ConvBlock(in_fts, out_fts[0], kernel_size=1),
            ConvBlock(out_fts[0], out_fts[1], kernel_size=3, padding=1)
        )
        self.subbranch1_1 = ConvBlock(out_fts[1], out_fts[1], kernel_size=(1, 3), padding=(0, 1))
        self.subbranch1_2 = ConvBlock(out_fts[1], out_fts[1], kernel_size=(3, 1), padding=(1, 0))

        self.branch2 = ConvBlock(in_fts, out_fts[2], kernel_size=1)
        self.subbranch2_1 = ConvBlock(out_fts[2], out_fts[2], kernel_size=(1, 3), padding=(0, 1))
        self.subbranch2_2 = ConvBlock(out_fts[2], out_fts[2], kernel_size=(3, 1), padding=(1, 0))

        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),  # or AVG
            ConvBlock(in_fts, out_fts[3], kernel_size=1)
        )
        self.branch4 = ConvBlock(in_fts, out_fts[4], kernel_size=1)

    def forward(self, input_img):
        o1 = self.branch1(input_img)
        o11 = self.subbranch1_1(o1)
        o12 = self.subbranch1_2(o1)
        o2 = self.branch2(input_img)
        o21 = self.subbranch2_1(o2)
        o22 = self.subbranch2_2(o2)
        o3 = self.branch3(input_img)
        o4 = self.branch4(input_img)
        return torch.cat([o11, o12, o21, o22, o3, o4], dim=1)


class AuxClassifier(nn.Module):
    def __init__(self, in_fts, num_classes):
        super(AuxClassifier, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size=(5, 5))
        self.conv = ConvBlock(in_fts, 128, kernel_size=1)
        # self.classifier = nn.Sequential(
        #     nn.Linear(5 * 5 * 128, 1024),
        #     nn.BatchNorm1d(num_features=1024),
        #     nn.Linear(1024, num_classes)
        # )
        self.conv2 = ConvBlock(128, 768, kernel_size=5)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        N = x.shape[0]
        x = self.pool(x)
        x = self.conv(x)
        x = self.conv2(x)
        x = x.reshape(N, -1)
        x = self.classifier(x)
        return x


class StemBlock(nn.Module):
    def __init__(self, in_channels):
        super(StemBlock, self).__init__()
        self.stem = nn.Sequential(
            ConvBlock(in_channels, 32, kernel_size=3, stride=2),
            ConvBlock(32, 32, kernel_size=3),
            ConvBlock(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            ConvBlock(64, 80, kernel_size=3),
            ConvBlock(80, 192, kernel_size=3, stride=2),
            ConvBlock(192, 288, kernel_size=3)
        )

    def forward(self, x):
        return self.stem(x)


class MyInceptionV3(nn.Module):
    def __init__(self, in_fts=3, num_classes=1000):
        super(MyInceptionV3, self).__init__()
        self.stem = StemBlock(in_fts)

        self.inceptX3 = nn.Sequential(InceptionX3(in_channels=288, inner=72),
                                      InceptionX3(in_channels=288, inner=72),
                                      InceptionX3(in_channels=288, inner=72))
        self.reduction1 = GridReduction(in_channels=288, out_channels=[96, 384])

        self.aux_classifier = AuxClassifier(768, num_classes)

        self.inceptX5 = nn.Sequential(InceptionX5(in_channels=768, inner=128),
                                      InceptionX5(in_channels=768, inner=128),
                                      InceptionX5(in_channels=768, inner=160),
                                      InceptionX5(in_channels=768, inner=160),
                                      InceptionX5(in_channels=768, inner=192))
        self.reduction2 = GridReduction(in_channels=768, out_channels=[192, 320])
        self.inceptX2 = nn.Sequential(InceptionX2(in_fts=1280, out_fts=[448, 384, 384, 192, 320]),
                                      InceptionX2(in_fts=2048, out_fts=[448, 384, 384, 192, 320]))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, input_img):
        bz = input_img.shape[0]
        x = self.stem(input_img)
        x = self.inceptX3(x)
        x = self.reduction1(x)
        aux_out = self.aux_classifier(x)
        x = self.inceptX5(x)
        x = self.reduction2(x)
        x = self.inceptX2(x)
        x = self.avgpool(x)
        x = x.reshape(bz, -1)
        x = self.fc(x)
        if self.training:
            return [x, aux_out]
        return x


def get_my_inception_v3_model(num_classes, device):
    return MyInceptionV3(num_classes=num_classes).to(device)


def train_my_inception_v3(dataloaders, image_datasets, num_classes, device, num_epochs=50):
    model = get_my_inception_v3_model(num_classes, device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.RMSprop(model.fc.parameters(), lr=0.045)
    early_stopping = EarlyStopping(model_name="MyInceptionV3_fruits299", save_best=True,
                                   use_early_stop=False)
    # torch.cuda.empty_cache()
    return train_model(model, dataloaders, image_datasets, criterion,
                       optimizer, device, early_stopping, num_epochs, has_aux=True)


def get_my_inception_v3_classifier(num_classes, device):
    from classifier import Classifier
    model = get_my_inception_v3_model(num_classes, device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # optimizer = optim.RMSprop(model.fc.parameters(), lr=0.045)
    optimizer = optim.Adam(model.fc.parameters())
    # optimizer = optim.RAdam(model.fc.parameters())
    return Classifier(model, optimizer, criterion, device, "MyInceptionV3_Adam", has_aux=True)
