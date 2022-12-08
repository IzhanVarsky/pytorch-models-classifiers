import math
from datetime import datetime

import torch
from matplotlib import pyplot as plt
from sklearn import metrics
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm


class EarlyStopping:
    def __init__(self, model_name, patience=15, min_delta=0,
                 save_best=False, use_early_stop=True, metric_decreasing=True):
        self.model_name = model_name
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_metric = None
        self.early_stop = False
        self.use_early_stop = use_early_stop
        self.save_best = save_best
        if metric_decreasing:
            self.is_cur_metric_better = lambda val: self.best_metric - val > self.min_delta
        else:
            self.is_cur_metric_better = lambda val: self.best_metric - val < self.min_delta

    def __call__(self, cur_metric, model):
        if self.best_metric == None:
            self.best_metric = cur_metric
            if self.save_best:
                self.save_best_model(model)
        elif self.is_cur_metric_better(cur_metric):
            self.best_metric = cur_metric
            self.counter = 0
            if self.save_best:
                self.save_best_model(model)
        else:
            self.counter += 1
            if self.use_early_stop and self.counter >= self.patience:
                self.early_stop = True

    def save_best_model(self, model):
        print("-" * 100)
        print(f">>> Saving the current {self.model_name} model with the best metric value...")
        torch.save(model.state_dict(), f'{self.model_name}_best_metric_model.pth')


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
                device, early_stopping, num_epochs=5, has_aux=False):
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
            y_test = []
            y_pred = []
            for inputs, labels in tqdm(dataloaders[phase]):

                inputs = inputs.to(device)
                labels = labels.to(device)

                if phase == 'train' and has_aux:
                    outputs, aux = model(inputs)
                    loss = criterion(outputs, labels) + 0.4 * criterion(aux, labels)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                y_test.extend(labels.cpu())
                y_pred.extend(preds.cpu())
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            f1_macro = metrics.f1_score(y_test, y_pred, average="macro")
            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])
            saved_epoch_losses[phase].append(epoch_loss)
            saved_epoch_accuracies[phase].append(epoch_acc.item())
            print('{} loss: {:.4f}, acc: {:.4f}, f1_macro: {:.4f}'
                  .format(phase, epoch_loss, epoch_acc, f1_macro))

        end_time = datetime.now()
        epoch_time = (end_time - start_time).total_seconds()
        print("-" * 100)
        print(f"Epoch Time: {math.floor(epoch_time // 60)}:{math.floor(epoch_time % 60)}")
        print("-" * 100)

        early_stopping(saved_epoch_losses['test'][-1], model)
        if early_stopping.early_stop:
            print('*** Early stopping ***')
            break
        if f1_macro > 0.95:
            print('*** Needed F1 macro achieved ***')
            break
    print("*** Training Completed ***")
    # plot_graphics(saved_epoch_losses, saved_epoch_accuracies)
    return model
