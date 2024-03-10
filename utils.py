import os
import shutil
import torch
import torchvision.transforms as transforms
import torchvision.datasets as torchdata
import torchvision.models as torchmodels
from data_loader import MIMIC

def save_args(args, cv_dir):
    """
    Save the command-line arguments to a text file.

    Args:
        args: argparse.Namespace, parsed command-line arguments.
        cv_dir (str): Path to the directory where the arguments will be saved.
    """
    shutil.copy(os.path.basename(__file__), cv_dir)
    with open(os.path.join(cv_dir, 'args.txt'), 'w') as f:
        f.write(str(args))

def get_transforms():
    """
    Define the transformation pipelines for training and testing.

    Returns:
        transform_train: torchvision.transforms.Compose, the training transformation pipeline.
        transform_test: torchvision.transforms.Compose, the testing transformation pipeline.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform_train = transforms.Compose([
       transforms.Resize(224),
       transforms.RandomCrop(224),
       transforms.ToTensor(),
       transforms.Normalize(mean, std)
    ])
    transform_test = transforms.Compose([
       transforms.Resize(224),
       transforms.CenterCrop(224),
       transforms.ToTensor(),
       transforms.Normalize(mean, std)
    ])

    return transform_train, transform_test

def get_dataset(train_csv, val_csv, pretrain=False):
    """
    Load the MIMIC dataset for training and validation.

    Args:
        train_csv (str): Path to the CSV file for the training set.
        val_csv (str): Path to the CSV file for the validation set.
        pretrain (bool, optional): If True, the dataset is used for pretraining. Defaults to False.

    Returns:
        trainset: MIMIC, the training dataset.
        testset: MIMIC, the validation dataset.
    """
    transform_train, transform_test = get_transforms()
    trainset = MIMIC(train_csv, transform_train)
    testset = MIMIC(val_csv, transform_test)

    return trainset, testset

def set_parameter_requires_grad(model, feature_extracting):
    """
    Set the requires_grad attribute of the model parameters.

    Args:
        model: torch.nn.Module, the model.
        feature_extracting (bool): If True, the parameters of the model will not be updated during training.
    """
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def get_model():
    """
    Get a pre-trained DenseNet model for feature extraction.

    Returns:
        dnet: torch.nn.Module, the DenseNet model.
    """
    dnet = torchmodels.densenet161(pretrained=True)
    set_parameter_requires_grad(dnet, False)
    num_ftrs = dnet.classifier.in_features
    dnet.classifier = torch.nn.Linear(num_ftrs, 300)

    return dnet
