import torch

from inception_v3.inception_v3_classifier import train_inception_v3, get_inception_v3_classifier
from inception_v3_from_scratch.my_inception import train_my_inception_v3, get_my_inception_v3_classifier
from resnet.resnet_classifier import train_resnet, get_resnet_classifier
from test_models import test_model
from utils import get_dataloader_dataset, EarlyStopping
from vgg.vgg_classifier import train_vgg, get_vgg11_classifier
from vit.vit_classifier import train_vit, get_vit_classifier


def run_train2():
    batch_size = 128
    train_dir = "./fruits-360_dataset_299/fruits-360/Training"
    test_dir = "./fruits-360_dataset_299/fruits-360/Test"
    dataset_name = "Fruits299"
    # train_dir = "./cars_archive_299/train"
    # test_dir = "./cars_archive_299/test"
    # dataset_name = "Cars299"

    dataloaders, image_datasets = get_dataloader_dataset(train_dir=train_dir,
                                                         test_dir=test_dir,
                                                         batch_size=batch_size)
    num_classes = len(image_datasets['train'].classes)
    print('Dataset Name:', dataset_name)
    print('Train Dataset Size:', len(image_datasets['train']))
    print('Test Dataset Size:', len(image_datasets['test']))
    print('batch_size:', batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classifiers = {"vit": get_vit_classifier,
                   "resnet": get_resnet_classifier,
                   "vgg11": get_vgg11_classifier,
                   "inception_v3": get_inception_v3_classifier,
                   "inception_v3_scratch": get_my_inception_v3_classifier,
                   }
    classifier = classifiers["inception_v3_scratch"](num_classes, device)
    checkpoint_name = f"{classifier.model_name}_{dataset_name}"
    early_stopping = EarlyStopping(model_name=checkpoint_name, save_best=True,
                                   use_early_stop=False, metric_decreasing=False)
    num_epochs = 100
    # classifier.load_checkpoint("InceptionTransfer_cars299_best_loss_model.pth")
    classifier.train_model(dataloaders, early_stopping, num_epochs=num_epochs)
    classifier.test_model(dataloaders)


def run_train():
    batch_size = 16
    # train_dir = "./fruits-360_dataset_299/fruits-360/Training"
    # test_dir = "./fruits-360_dataset_299/fruits-360/Test"
    # dataset_name = "Fruits299"
    train_dir = "./cars_archive_299/train"
    test_dir = "./cars_archive_299/test"
    dataset_name = "Cars299"

    dataloaders, image_datasets = get_dataloader_dataset(train_dir=train_dir,
                                                         test_dir=test_dir,
                                                         batch_size=batch_size)
    num_classes = len(image_datasets['train'].classes)
    print('Dataset Name:', dataset_name)
    print('Train Dataset Size:', len(image_datasets['train']))
    print('Test Dataset Size:', len(image_datasets['test']))
    print('batch_size:', batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainers = {"vit": train_vit,
                "resnet": train_resnet,
                "vgg": train_vgg,
                "inception_v3": train_inception_v3,
                "inception_v3_scratch": train_my_inception_v3,
                }
    model = trainers["inception_v3"](dataloaders, image_datasets, num_classes, device)
    test_model(model, dataloaders, device)


if __name__ == "__main__":
    # run_train()
    run_train2()
