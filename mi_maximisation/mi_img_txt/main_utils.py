import os
from tqdm import tqdm, trange
import logging
import numpy as np
import sklearn
import time

import torch
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule

from .model import build_bert_model, build_resnet_model
from .model import ImageReportModel
from .model import make_mlp
from .utils import MimicID
from .model_utils import CXRImageReportDataset
from .mi_critics import dv_bound_loss, infonce_bound_loss


def build_training_imagereportset(text_token_features, img_dir, img_size: int, 
                                  dataset_metadata='../data/training.csv',
                                  random_degrees=[-20,20], random_translate=[0.1,0.1]):
    """ Build a image-report dataset for model training 
        with data augmentation on the images on the fly
        
    Args:
        text_token_features (list): List of text token features.
        img_dir (str): Directory containing the images.
        img_size (int): Size of the images.
        dataset_metadata (str): Path to the dataset metadata.
        random_degrees (list): Range of degrees for random image rotation.
        random_translate (list): Range of translation for random image translation.
        
    Returns:
        CXRImageReportDataset: Training dataset.
    """

    # Define transformations for data augmentation
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda img: img.astype(np.int16)),
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.RandomAffine(degrees=random_degrees, translate=random_translate),
        torchvision.transforms.CenterCrop(img_size),
        torchvision.transforms.Lambda(lambda img: np.array(img).astype(np.float32)),
        torchvision.transforms.Lambda(lambda img: img / max(1e-3, img.max()))
    ])

    # Build the training dataset
    training_dataset = CXRImageReportDataset(text_token_features=text_token_features,
                                             img_dir=img_dir, 
                                             dataset_metadata=dataset_metadata, 
                                             transform=transform)

    return training_dataset


class ImageTextModelManager:
    """ A manager class that creates and manages the joint image-text model
        with global mutual information criterion 
    """

    def __init__(self, bert_pretrained_dir, bert_config_name,
                 output_channels, image_model_name):
        """ Initialize the ImageTextModelManager.

        Args:
            bert_pretrained_dir (str): Directory containing the pretrained BERT model.
            bert_config_name (str): Name of the BERT configuration.
            output_channels (int): Number of output channels.
            image_model_name (str): Name of the image model.
        """
        self.bert_pretrained_dir = bert_pretrained_dir
        self.bert_config_name = bert_config_name
        self.output_channels = output_channels
        self.image_model_name = image_model_name

        # Build the BERT-based text model
        self.text_model, self.bert_config = \
            build_bert_model(bert_pretrained_dir=bert_pretrained_dir,
                             bert_config_name=bert_config_name,
                             output_channels=output_channels)

        # Build the ResNet-based image model
        self.image_model = build_resnet_model(model_name=image_model_name, 
                                              output_channels=output_channels)

        # Initialize the joint image-text model
        self.model = ImageReportModel(text_model=self.text_model,
                                      bert_config=self.bert_config,
                                      image_model=self.image_model)

        # Initialize the mutual information discriminator
        self.mi_discriminator = make_mlp(1536, [1024, 512])
        self.logger = logging.getLogger(__name__)

    def create_mi_pairs(self, embedding_img, embedding_txt, study_id: list, device):
        """ Concatenate image and text features to create pairs for mutual information estimation.

        Args:
            embedding_img (torch.Tensor): Image embeddings.
            embedding_txt (torch.Tensor): Text embeddings.
            study_id (list): List of study IDs.
            device (torch.device): Device to perform computations on.

        Returns:
            torch.Tensor: Concatenated feature pairs.
        """
        batch_size = len(study_id)

        # Concatenate matched/positive pairs
        mi_input = torch.cat((embedding_img, embedding_txt), 1)

        # Shuffle and concatenate unmatched/negative pairs
        for gap in range(batch_size-1):
            for i in range(batch_size):
                if i+(gap+1)<batch_size:
                    j = i+(gap+1) 
                else:
                    j = i+(gap+1) - batch_size
                if study_id[i] != study_id[j]:
                    embedding_cat = torch.cat((embedding_img[i], embedding_txt[j]))
                    embedding_cat = torch.reshape(embedding_cat, (1, embedding_cat.shape[0]))
                    mi_input = torch.cat((mi_input, embedding_cat), 0)

        return mi_input

    def train(self, text_token_features, device, args):
        """ Train the joint image-text model.

        Args:
            text_token_features (list): List of text token features.
            device (torch.device): Device to perform computations on.
            args (argparse.Namespace): Parsed command-line arguments.

        Returns:
            None
        """
        logger = logging.getLogger(__name__)

        # Build the training dataset
        dataset = build_training_imagereportset(text_token_features=text_token_features,
                                                img_dir=args.image_dir,
                                                img_size=args.img_size,
                                                dataset_metadata=args.dataset_metadata)
        data_loader = DataLoader(dataset, batch_size=args.batch_size,
                                 shuffle=True, num_workers=8,
                                 pin_memory=True, drop_last=True)
        logger.info(f'Total number of training image-report pairs: {len(dataset)}')

        # Move models to device
        self.model = self.model.to(device)
        self.mi_discriminator = self.mi_discriminator.to(device)

        # Define the mutual information loss criterion
        if args.mi_estimator == 'dv':
            mi_critic = dv_bound_loss
        if args.mi_estimator == 'infonce':
            mi_critic = infonce_bound_loss

        # Define optimizers and learning rate scheduler
        img_optimizer = optim.Adam(self.model.image_model.parameters(), lr=args.init_lr)
        mi_optimizer = optim.Adam(self.mi_discriminator.parameters(), lr=args.init_lr)

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        param_txt = list(self.model.text_model.named_parameters())
        grouped_parameters_txt = [
            {'params': [p for n, p in param_txt if not any(nd in n for nd in no_decay)], 
            'weight_decay': 0.1},
            {'params': [p for n, p in param_txt if any(nd in n for nd in no_decay)], 
            'weight_decay': 0.0}
            ]
        txt_optimizer = AdamW(grouped_parameters_txt, 
                              lr=2e-5,
                              correct_bias=False)
        num_train_steps = int(args.num_train_epochs*len(data_loader))
        scheduler = WarmupLinearSchedule(txt_optimizer, 
                                         warmup_steps=0.1*num_train_steps,
                                         t_total=num_train_steps)

        # Train the model
        self.model.train()
        total_steps = 0
