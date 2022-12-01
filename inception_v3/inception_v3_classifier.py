from torch import nn, optim
from torchvision import models

from utils import EarlyStopping, train_model


def get_inception_v3_model(num_classes, device, pretrained):
    model = models.inception_v3(pretrained=pretrained).to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(2048, num_classes).to(device)
    model.aux_logits = False
    return model


def train_inception_v3(dataloaders, image_datasets, num_classes, device, num_epochs=50):
    model = get_inception_v3_model(num_classes, device, pretrained=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.fc.parameters(), lr=0.045)
    early_stopping = EarlyStopping(model_name="InceptionTransfer_fruits299", save_best=True,
                                   use_early_stop=False)
    # torch.cuda.empty_cache()
    return train_model(model, dataloaders, image_datasets, criterion,
                       optimizer, device, early_stopping, num_epochs)
