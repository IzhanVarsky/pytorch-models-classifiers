import torch

from inception_v3.inception_v3_classifier import train_inception_v3
from resnet.resnet_classifier import train_resnet
from test_models import test_model
from utils import get_dataloader_dataset
from vgg.vgg_classifier import train_vgg
from vit.vit_classifier import train_vit


def run_train():
    batch_size = 64
    train_dir = "./fruits-360_dataset_299/fruits-360/Training"
    test_dir = "./fruits-360_dataset_299/fruits-360/Test"
    # train_dir = "./cars_archive_299/train"
    # test_dir = "./cars_archive_299/test"

    dataloaders, image_datasets = get_dataloader_dataset(train_dir=train_dir,
                                                         test_dir=test_dir,
                                                         batch_size=batch_size)
    num_classes = len(image_datasets['train'].classes)
    print('Train Dataset Size: ', len(image_datasets['train']))
    print('Test Dataset Size: ', len(image_datasets['test']))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainers = {"vit": train_vit,
                "resnet": train_resnet,
                "vgg": train_vgg,
                "inception_v3": train_inception_v3,
                }
    model = trainers["inception_v3"](dataloaders, image_datasets, num_classes, device)
    test_model(model, dataloaders, device)


if __name__ == "__main__":
    run_train()
