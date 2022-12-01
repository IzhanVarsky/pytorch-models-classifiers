import torch
import torch.nn as nn
from tqdm import tqdm

from resnet.resnet_classifier import get_resnet_model
from utils import get_dataloader_dataset
from sklearn.metrics import f1_score

from vgg.vgg_classifier import get_vgg_model
from vit.vit_classifier import get_vit_model


def test_model(model, dataloaders, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    # total_f1 = 0
    total_acc = 0
    steps_count = 0

    all_labels = []
    all_predicts = []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloaders['test']):
            # print("-" * 10)
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()

            total_acc += correct / len(predicted)
            # print(f"Correct: {correct}/{len(predicted)}")
            # print(f"Incorrect: {(predicted != labels).sum().item()}/{len(predicted)}")
            all_labels.extend(labels.cpu())
            all_predicts.extend(predicted.cpu())
            # f1 = f1_score(labels.cpu(), predicted.cpu(), average='macro')
            # total_f1 += f1
            # print("F1 score:", f1)

            loss = criterion(outputs, labels).item()
            total_loss += loss
            # print("Loss:", loss)

            steps_count += 1
            # if steps_count == max_test_cnt:
            #     break
    # print("=" * 20)
    total_f1 = f1_score(all_labels, all_predicts, average="macro")
    print("Mean total loss:", total_loss / steps_count)
    print("Mean total accuracy:", total_acc / steps_count)
    # print("Mean total F1_macro score:", total_f1 / steps_count)
    print("Mean total F1_macro score:", total_f1)


def test_from_checkpoint(model, checkpoint_path, dataloaders, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    test_model(model, dataloaders, device)


def test_vgg(dataloaders, num_classes, device):
    print("*" * 25)
    print(">> Testing VGG network")
    model = get_vgg_model(num_classes, device)
    # test_from_checkpoint(model, 'VGG11_best_loss_model.pth', dataloaders, device)
    test_from_checkpoint(model, 'VGG11_lr0.0001_best_loss_model.pth', dataloaders, device)


def test_resnet(dataloaders, num_classes, device):
    print("*" * 25)
    print(">> Testing ResNet network")
    model = get_resnet_model(num_classes, device, pretrained=False)
    # 3 epochs model
    ckpt_path = 'ResNetTransfer_best_loss_model_99perc.pth'
    test_from_checkpoint(model, ckpt_path, dataloaders, device)


def test_vit(dataloaders, num_classes, device):
    print("*" * 25)
    print(">> Testing ViT network")
    model = get_vit_model(num_classes, num_layers=8, device=device)
    # 9 epochs model
    ckpt_path = 'VisionTransformer_best_loss_model_8layers_100percent_lr1e-5.pth'
    test_from_checkpoint(model, ckpt_path,
                         dataloaders, device)


if __name__ == "__main__":
    dataloaders, image_datasets = get_dataloader_dataset(need_train=False, need_test=True,
                                                         batch_size=64,
                                                         num_workers=2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_classes = len(image_datasets['test'].classes)
    test_vgg(dataloaders, num_classes, device)
    test_resnet(dataloaders, num_classes, device)
    test_vit(dataloaders, num_classes, device)
