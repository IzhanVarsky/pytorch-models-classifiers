import math
from datetime import datetime

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm


class EarlyStopping:
    def __init__(self, model_name, patience=15, min_delta=0, save_best=False):
        self.model_name = model_name
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.save_best = save_best

    def __call__(self, val_loss, model):
        if self.best_loss == None:
            self.best_loss = val_loss
            if self.save_best:
                self.save_best_model(model)
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.save_best:
                self.save_best_model(model)
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_best_model(self, model):
        print(f">>> Saving the current {self.model_name} model with the best loss value...")
        print("-" * 100)
        torch.save(model.state_dict(), f'{self.model_name}_best_loss_model.pth')


def plot_graphics(losses, accuracies):
    fig, ax = plt.subplots(1, 2, figsize=(8, 4), dpi=120)

    # Loss
    ax[0].plot(losses['train'], label='Training Loss')
    ax[0].plot(losses['test'], label='Testing Loss')
    ax[0].axis(ymin=-0.10, ymax=10)
    ax[0].set_title('Loss Plot')
    ax[0].legend()

    # Accuracy
    ax[1].plot(accuracies['train'], label='Training Accuracy')
    ax[1].plot(accuracies['test'], label='Testing Accuracy')
    ax[1].axis(ymin=0, ymax=101)
    ax[1].set_title('Accuracy Plot')
    ax[1].legend()
    plt.show()


def get_dataloader_dataset(train_dir="",
                           test_dir="",
                           need_train=True,
                           need_test=True,
                           batch_size=64,
                           num_workers=8):
    image_datasets = dict()
    dataloaders = dict()
    if need_train:
        image_datasets['train'] = ImageFolder(train_dir, transform=transforms.ToTensor())
        dataloaders['train'] = DataLoader(image_datasets['train'],
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=num_workers,
                                          pin_memory=True)
    if need_test:
        image_datasets['test'] = ImageFolder(test_dir, transform=transforms.ToTensor())
        dataloaders['test'] = DataLoader(image_datasets['test'],
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=num_workers,
                                         pin_memory=True)
    if need_test and need_train:
        assert image_datasets['train'].classes == image_datasets['test'].classes
    return dataloaders, image_datasets


def train_model(model, dataloaders, image_datasets, criterion, optimizer,
                device, early_stopping, num_epochs=5):
    saved_epoch_losses = {'train': [], 'test': []}
    saved_epoch_accuracies = {'train': [], 'test': []}

    for epoch in range(num_epochs):
        start_time = datetime.now()

        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'test']:
            print("--- Cur phase:", phase)
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase]):

                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])
            saved_epoch_losses[phase].append(epoch_loss)
            saved_epoch_accuracies[phase].append(epoch_acc.item())
            print('{} loss: {:.4f}, acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
        end_time = datetime.now()
        epoch_time = (end_time - start_time).total_seconds()
        print("-" * 100)
        print(f"Epoch Time: {math.floor(epoch_time // 60)}:{math.floor(epoch_time % 60)}")
        print("-" * 100)

        early_stopping(saved_epoch_losses['test'][-1], model)
        if early_stopping.early_stop:
            print('*** Early stopping ***')
            break
    print("*** Training Completed ***")
    # plot_graphics(saved_epoch_losses, saved_epoch_accuracies)
    return model
