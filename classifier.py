import math
from datetime import datetime

import torch
from sklearn import metrics
from torch import nn
from tqdm import tqdm


class Classifier(nn.Module):
    def __init__(self, model, optim, criterion, device, model_name, has_aux=False):
        super().__init__()
        self.model = model.to(device)
        self.optimizer = optim
        self.criterion = criterion
        self.device = device
        self.model_name = model_name
        self.has_aux = has_aux

    def load_checkpoint(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)

    def run_epoch(self, phase, dataloader):
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()

        running_loss = 0.0
        running_corrects = 0
        y_test = []
        y_pred = []
        all_elems_count = 0
        for inputs, labels in tqdm(dataloader):
            bz = inputs.shape[0]
            all_elems_count += bz

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            if phase == 'train' and self.has_aux:
                outputs, aux = self.model(inputs)
                loss = self.criterion(outputs, labels) + 0.4 * self.criterion(aux, labels)
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            _, preds = torch.max(outputs, 1)
            y_test.extend(labels.cpu())
            y_pred.extend(preds.cpu())
            running_loss += loss.item() * bz
            running_corrects += torch.sum(preds == labels.data)

        f1_macro = metrics.f1_score(y_test, y_pred, average="macro")
        epoch_loss = running_loss / all_elems_count
        epoch_acc = running_corrects.double().item() / all_elems_count
        return epoch_loss, epoch_acc, f1_macro

    def test_epoch(self, dataloader):
        with torch.no_grad():
            return self.run_epoch('test', dataloader)

    def train_epoch(self, dataloader):
        return self.run_epoch('train', dataloader)

    def train_model(self, dataloaders, early_stopping, num_epochs=5):
        print(f"Training model {self.model_name} with params:")
        print(f"Optim: {self.optimizer}")
        print(f"Criterion: {self.criterion}")
        
        saved_epoch_losses = {'train': [], 'test': []}
        saved_epoch_accuracies = {'train': [], 'test': []}
        saved_epoch_f1_macros = {'train': [], 'test': []}

        # save_by = saved_epoch_losses
        # save_by = saved_epoch_accuracies
        save_by = saved_epoch_f1_macros

        for epoch in range(num_epochs):
            start_time = datetime.now()

            print("=" * 100)
            print('Epoch {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 10)

            for phase in ['train', 'test']:
                print("--- Cur phase:", phase)
                epoch_loss, epoch_acc, f1_macro = self.train_epoch(dataloaders[phase]) if phase == 'train' \
                    else self.test_epoch(dataloaders[phase])
                saved_epoch_losses[phase].append(epoch_loss)
                saved_epoch_accuracies[phase].append(epoch_acc)
                saved_epoch_f1_macros[phase].append(f1_macro)
                print('{} loss: {:.4f}, acc: {:.4f}, f1_macro: {:.4f}'
                      .format(phase, epoch_loss, epoch_acc, f1_macro))

            end_time = datetime.now()
            epoch_time = (end_time - start_time).total_seconds()
            print("-" * 10)
            print(f"Epoch Time: {math.floor(epoch_time // 60)}:{math.floor(epoch_time % 60)}")

            early_stopping(save_by['test'][-1], self.model)
            if early_stopping.early_stop:
                print('*** Early stopping ***')
                break
            if f1_macro > 0.95:
                print('*** Needed F1 macro achieved ***')
                break
        print("*** Training Completed ***")
        # plot_graphics(saved_epoch_losses, saved_epoch_accuracies)
        return self.model

    def test_model(self, dataloaders):
        print("*" * 25)
        print(f">> Testing {self.model_name} network")
        epoch_loss, epoch_acc, f1_macro = self.test_epoch(dataloaders['test'])
        print("Mean total loss:", epoch_loss)
        print("Mean total accuracy:", epoch_acc)
        print("Mean total F1_macro score:", f1_macro)
