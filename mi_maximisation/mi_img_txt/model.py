import csv
import os
import numpy as np
from math import floor, ceil
import json
import logging

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from pytorch_transformers import BertModel, BertConfig
from pytorch_transformers.modeling_bert import BertPreTrainedModel


def make_mlp(input_dim, hidden_dims: list, output_dim=1, activation='relu'):
    """
    Create a multi-layer perceptron (MLP) neural network.

    Args:
    - input_dim (int): Dimensionality of the input features.
    - hidden_dims (list): List of integers specifying the number of neurons in each hidden layer.
    - output_dim (int): Dimensionality of the output.
    - activation (str): Activation function to be used in hidden layers. Default is 'relu'.

    Returns:
    - nn.Sequential: Sequential container of the MLP layers.

    """
    # Mapping activation function names to PyTorch activation modules
    activation_map = {
        'relu': nn.ReLU
    }

    # Retrieve the activation module corresponding to the specified activation function
    activation_module = activation_map[activation]

    # Number of hidden layers in the MLP
    num_hidden_layers = len(hidden_dims)

    # Initialize an empty list to store layers of the MLP
    seq = []

    # Add the input layer and the first hidden layer
    seq += [nn.Linear(input_dim, hidden_dims[0]), activation_module()]

    # Add remaining hidden layers
    for i in range(num_hidden_layers-1):
        # Add a linear layer followed by activation function for each hidden layer
        seq += [nn.Linear(hidden_dims[i], hidden_dims[i+1]), activation_module()]

    # Add the output layer
    seq += [nn.Linear(hidden_dims[-1], output_dim)]

    # Create a sequential container and return
    return nn.Sequential(*seq)

# Adapted from
# https://medium.com/huggingface/multi-label-text-classification-using-bert-the-mighty-transformer-69714fa3fb3d
class TextBert(BertPreTrainedModel):
    """
    Text global feature embedder implemented with a BERT model.

    Args:
    - config (BertConfig): Configuration for the BERT model.

    """

    def __init__(self, config):
        super(TextBert, self).__init__(config)
        self.num_classes = config.num_classes

        # BERT model
        self.bert = BertModel(config)
        # Dropout layer
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # Classifier layer
        self.classifier = nn.Linear(config.hidden_size, self.config.num_classes)

        # Initialize weights
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        """
        Forward pass of the TextBert model.

        Args:
        - input_ids (torch.Tensor): Input token ids.
        - token_type_ids (torch.Tensor): Segment token ids (optional).
        - attention_mask (torch.Tensor): Attention mask (optional).
        - labels (torch.Tensor): Labels (optional).

        Returns:
        - outputs (tuple): Tuple containing pooled_output, logits, and optionally hidden states and attentions.

        """
        # Forward pass through the BERT model
        outputs = self.bert(input_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask)
        
        # Extract pooled_output (CLS token representation)
        pooled_output = outputs[1]
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        # Pass through classifier layer
        logits = self.classifier(pooled_output)
        # Aggregate outputs
        outputs = (pooled_output, logits,) + outputs[2:]

        return outputs

    def freeze_bert_encoder(self):
        """Freezes the parameters of the BERT encoder."""
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        """Unfreezes the parameters of the BERT encoder."""
        for param in self.bert.parameters():
            param.requires_grad = True



def build_bert_model(bert_pretrained_dir, bert_config_name, output_channels):
    """
    Build a BERT-based text classification model.

    Args:
    - bert_pretrained_dir (str): Directory containing pretrained BERT model.
    - bert_config_name (str): Name of the BERT configuration file.
    - output_channels (int): Number of output channels.

    Returns:
    - bertgb_model (TextBert): BERT-based text classification model.
    - config (BertConfig): BERT model configuration.

    """
    # Load BERT configuration from file
    config_path = os.path.join(bert_pretrained_dir, bert_config_name)
    with open(config_path) as f:
        print('BERT config:', json.load(f))
    config = BertConfig.from_json_file(config_path)
    config.num_classes = output_channels

    # Initialize BERT-based text classification model
    bertgb_model = TextBert.from_pretrained(bert_pretrained_dir,
                                             config=config)

    return bertgb_model, config


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """
    3x3 convolution with padding.

    Args:
    - in_planes (int): Number of input channels.
    - out_planes (int): Number of output channels.
    - stride (int): Stride of the convolution operation. Default is 1.
    - groups (int): Number of groups for grouped convolution. Default is 1.
    - dilation (int): Dilation rate of the convolution operation. Default is 1.

    Returns:
    - nn.Conv2d: 3x3 convolutional layer.

    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=dilation,
                     groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """
    1x1 convolution.

    Args:
    - in_planes (int): Number of input channels.
    - out_planes (int): Number of output channels.
    - stride (int): Stride of the convolution operation. Default is 1.

    Returns:
    - nn.Conv2d: 1x1 convolutional layer.

    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1,
                     stride=stride, bias=False)


class BasicBlock(nn.Module):
    """
    Basic building block for ResNet.

    Args:
    - inplanes (int): Number of input channels.
    - planes (int): Number of output channels.
    - stride (int): Stride for the first convolutional layer. Default is 1.
    - downsample (nn.Module): Downsample function for residual connection. Default is None.
    - norm_layer (nn.Module): Normalization layer. Default is nn.BatchNorm2d.

    """

    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1,
                 downsample=None,
                 norm_layer=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()

        # First convolutional layer
        self.conv1 = conv3x3(inplanes, planes, stride)
        # Batch normalization
        self.bn1 = norm_layer(planes)
        # ReLU activation function
        self.relu = nn.ReLU(inplace=True)
        # Second convolutional layer
        self.conv2 = conv3x3(planes, planes)
        # Batch normalization
        self.bn2 = norm_layer(planes)
        # Downsample function for residual connection
        self.downsample = downsample
        # Stride for the convolutional layers
        self.stride = stride

    def forward(self, x):
        """
        Forward pass of the BasicBlock.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - out (torch.Tensor): Output tensor.

        """
        # Save the input for the residual connection
        identity = x

        # First convolutional layer
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second convolutional layer
        out = self.conv2(out)
        out = self.bn2(out)

        # Apply downsampling if necessary
        if self.downsample is not None:
            identity = self.downsample(x)

        # Add the residual connection
        out += identity
        # Apply ReLU activation
        out = self.relu(out)

        return out

class ResNet256_6_2_1(nn.Module):
    """A residual network with 6 residual layers, 2x2 average pooling, and 1 fully connected layer.
    
    This class defines a ResNet with 6 residual layers, where each residual layer is comprised of
    multiple residual blocks. The network architecture also includes average pooling and a fully 
    connected layer for classification.

    Args:
    - block (nn.Module): Type of residual block (e.g., BasicBlock or Bottleneck).
    - blocks_per_layers (list): Number of residual blocks per layer.
    - output_channels (int): Number of output channels for the final fully connected layer.
    - norm_layer (nn.Module): Normalization layer to be used.
    - zero_init_residual (bool): Whether to initialize residual branches with zeros.

    """
    def __init__(self, block, blocks_per_layers, output_channels=4, 
                 norm_layer=nn.BatchNorm2d, zero_init_residual=False):
        """Initialize the ResNet256_6_2_1 model.

        Args:
        - block (nn.Module): Type of residual block (e.g., BasicBlock or Bottleneck).
        - blocks_per_layers (list): Number of residual blocks per layer.
        - output_channels (int): Number of output channels for the final fully connected layer.
        - norm_layer (nn.Module): Normalization layer to be used.
        - zero_init_residual (bool): Whether to initialize residual branches with zeros.

        """
        super(ResNet256_6_2_1, self).__init__()

        self._norm_layer = norm_layer

        self.inplanes = 8
        self.dilation = 1

        # Initial convolution layer
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=3, stride=1, 
                               padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual layers
        self.layer1 = self._make_layer(block, 8, blocks_per_layers[0], stride=2)
        self.layer2 = self._make_layer(block, 16, blocks_per_layers[1], stride=2)
        self.layer3 = self._make_layer(block, 32, blocks_per_layers[2], stride=2)
        self.layer4 = self._make_layer(block, 64, blocks_per_layers[3], stride=2)
        self.layer5 = self._make_layer(block, 128, blocks_per_layers[4], stride=2)
        self.layer6 = self._make_layer(block, 192, blocks_per_layers[5], stride=2)

        # Average pooling and fully connected layer
        self.avgpool = nn.AvgPool2d((2, 2))
        self.fc1 = nn.Linear(768, output_channels)  # 768 is the output size after avgpool
        self.softmax = nn.Softmax(dim=1)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_of_blocks, stride=1):
        """Create a residual layer consisting of multiple residual blocks.

        Args:
        - block (nn.Module): Type of residual block (e.g., BasicBlock or Bottleneck).
        - planes (int): Number of input and output channels for the layer.
        - num_of_blocks (int): Number of residual blocks in the layer.
        - stride (int): Stride for the first residual block.

        Returns:
        - nn.Sequential: Sequential container of the residual blocks.

        """
        norm_layer = self._norm_layer

        # Downsample layer for residual connection
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes, stride),
                                       norm_layer(planes))

        # Create residual blocks
        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample,
                            norm_layer=norm_layer))
        self.inplanes = planes
        for _ in range(1, num_of_blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass of the ResNet model.

        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, 1, 256, 256).

        Returns:
        - tuple: Tuple containing output probabilities, global features, local features, and logits.

        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        z_local = self.layer5(x)
        x = self.layer6(z_local)
        
        x = self.avgpool(x)
        z = torch.flatten(x, 1)
        y_logits = self.fc1(z)
        y = self.softmax(y_logits)

        return y, z, z_local, y_logits

    # based on 
    # https://github.com/huggingface/transformers/blob/v1.0.0/pytorch_transformers/modeling_utils.py
    def save_pretrained(self, save_directory, epoch=-1):
        """
        Save the model and its configuration to a directory.

        Args:
        - save_directory (str): Directory path where the model and configuration will be saved.
        - epoch (int): Epoch number to include in the filename. Default is -1, which means no epoch number.

        """
        # Ensure that the save directory exists; if not, create it
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        assert os.path.isdir(save_directory)

        # If using distributed training, only save the model itself
        model_to_save = self.module if hasattr(self, 'module') else self

        # Define the output file path for the model checkpoint
        if epoch == -1:
            output_model_file = os.path.join(save_directory, 'pytorch_model.bin')
        else:
            output_model_file = os.path.join(save_directory, 'pytorch_model_epoch'+str(epoch)+'.bin')

        # Save the model state_dict to the output file
        torch.save(model_to_save.state_dict(), output_model_file)

    # based on 
    # https://github.com/huggingface/transformers/blob/v1.0.0/pytorch_transformers/modeling_utils.py
    @classmethod
    def from_pretrained(cls, pretrained_model_path, block, blocks_per_layers, 
                        output_channels, loading_from_joint=False, freeze_encoder=False,
                        *inputs, **kwargs):
        """
        Instantiate a model and load pretrained weights from a checkpoint file.

        Args:
        - pretrained_model_path (str): Path to the pretrained model checkpoint.
        - block (nn.Module): ResNet block type (e.g., BasicBlock).
        - blocks_per_layers (list): Number of blocks per layer.
        - output_channels (int): Number of output channels.
        - loading_from_joint (bool): Whether loading from a joint model checkpoint.
        - freeze_encoder (bool): Whether to freeze encoder layers.
        - *inputs: Additional positional arguments.
        - **kwargs: Additional keyword arguments.

        Returns:
        - model: Instantiated and pretrained ResNet model.

        """
        logger = logging.getLogger(__name__)

        # Pop out special keyword arguments for compatibility with PyTorch's loading function
        state_dict = kwargs.pop('state_dict', None)
        output_loading_info = kwargs.pop('output_loading_info', False)

        # Instantiate the model
        model = cls(block, blocks_per_layers, output_channels=output_channels, **kwargs)

        # If no specific state_dict provided, load from the specified checkpoint file
        if state_dict is None:
            state_dict = torch.load(pretrained_model_path, map_location='cpu')

        # Convert old format to new format if needed from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)
        
        # Only keep parameters of keys with 'image_model.*' if loading from a joint model checkpoint
        if loading_from_joint:
            old_keys = []
            new_keys = []
            for key in state_dict.keys():
                if 'image_model.' == key[:12]:
                    old_keys.append(key)
                    new_keys.append(key[12:])
            for old_key, new_key in zip(old_keys, new_keys):
                if 'image_model.fc' in old_key:
                    state_dict.pop(old_key, None)
                else:
                    state_dict[new_key] = state_dict.pop(old_key)

        # Load weights from the state_dict
        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        # Copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(model)

        # Print loading information if requested
        if output_loading_info:
            if len(missing_keys) > 0:
                logger.info("Weights of {} not initialized from pretrained model: {}".format(
                    model.__class__.__name__, missing_keys))
            if len(unexpected_keys) > 0:
                logger.info("Weights from pretrained model not used in {}: {}".format(
                    model.__class__.__name__, unexpected_keys))
            if len(error_msgs) > 0:
                raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                                   model.__class__.__name__, "\n\t".join(error_msgs)))

        # Freeze encoder layers if requested
        if freeze_encoder:
            for n, p in model.named_parameters():
                if not ('layer6' in n) and not ('fc' in n): 
                    p.requires_grad = False

        return model


def build_resnet256_6_2_1(block=BasicBlock, blocks_per_layers=[2, 2, 2, 2, 2, 2], 
                          pretrained=False, pretrained_model_path=None, output_channels=4, 
                          loading_from_joint=False, freeze_encoder=False, **kwargs):
    """
    Build ResNet256_6_2_1 model with specified configurations.

    Args:
    - block (nn.Module): ResNet block type (e.g., BasicBlock).
    - blocks_per_layers (list): Number of blocks per layer.
    - pretrained (bool): Whether to load pretrained weights.
    - pretrained_model_path (str): Path to the pretrained model checkpoint.
    - output_channels (int): Number of output channels.
    - loading_from_joint (bool): Whether loading from a joint model checkpoint.
    - freeze_encoder (bool): Whether to freeze encoder layers.
    - **kwargs: Additional keyword arguments.

    Returns:
    - model: ResNet256_6_2_1 model instance.

    """
    # Build the ResNet256_6_2_1 model
    model = ResNet256_6_2_1(block, blocks_per_layers, output_channels=output_channels, **kwargs)
    
    # Load pretrained weights if specified
    if pretrained:
        model = model.from_pretrained(pretrained_model_path, block, blocks_per_layers,
                                      output_channels, loading_from_joint=loading_from_joint, 
                                      freeze_encoder=freeze_encoder, **kwargs)
    return model


def build_resnet_model(model_name, checkpoint_path=None, output_channels=4, 
                       loading_from_joint=False, freeze_encoder=False):
    """
    Build a ResNet model according to the specified architecture and options.

    Args:
    - model_name (str): Name of the ResNet model architecture.
    - checkpoint_path (str): Path to the checkpoint file if pretrained weights are to be loaded.
    - output_channels (int): Number of output channels.
    - loading_from_joint (bool): Flag indicating whether loading from a joint model checkpoint.
    - freeze_encoder (bool): Flag indicating whether to freeze the encoder layers.

    Returns:
    - model: ResNet model instance.

    """
    if checkpoint_path is None:
        if model_name == 'resnet256_6_2_1':
            # Build ResNet256_6_2_1 model from scratch
            model = build_resnet256_6_2_1(output_channels=output_channels)
    else:
        if model_name == 'resnet256_6_2_1':
            # Build ResNet256_6_2_1 model from a pretrained checkpoint
            model = build_resnet256_6_2_1(output_channels=output_channels,
                                          pretrained=True,
                                          pretrained_model_path=checkpoint_path,
                                          loading_from_joint=loading_from_joint,
                                          freeze_encoder=freeze_encoder)
    return model

class ImageReportModel(nn.Module):
    """A joint image-report model combining text and image features.

    This model combines features extracted from a text model and an image model
    to produce joint embeddings and logits.

    Args:
    - text_model (nn.Module): Pre-trained text model.
    - bert_config (BertConfig): Configuration object for BERT model.
    - image_model (nn.Module): Pre-trained image model.

    """
    def __init__(self, text_model, bert_config, image_model):
        """Initialize the ImageReportModel.

        Args:
        - text_model (nn.Module): Pre-trained text model.
        - bert_config (BertConfig): Configuration object for BERT model.
        - image_model (nn.Module): Pre-trained image model.

        """
        super(ImageReportModel, self).__init__()
        self.text_model = text_model
        self.bert_config = bert_config
        self.image_model = image_model

    def forward(self, img, txt_ids, txt_masks=None, txt_segments=None):
        """Forward pass of the ImageReportModel.

        Args:
        - img (torch.Tensor): Input image tensor.
        - txt_ids (torch.Tensor): Input text token IDs.
        - txt_masks (torch.Tensor): Attention masks for text.
        - txt_segments (torch.Tensor): Segment IDs for text.

        Returns:
        - tuple: Tuple containing joint image embedding, joint text embedding, image logits, and text logits.

        """
        # Forward pass through image model
        outputs_img = self.image_model.forward(img)
        embedding_img = outputs_img[1]  # Extract image embedding
        logits_img = outputs_img[-1]    # Extract image logits

        # Forward pass through text model
        inputs_txt = {'input_ids': txt_ids,
                      'attention_mask': txt_masks,
                      'token_type_ids': txt_segments}
        outputs_txt = self.text_model.forward(**inputs_txt)
        embedding_txt = outputs_txt[0]  # Extract text embedding
        logits_txt = outputs_txt[1]      # Extract text logits

        return embedding_img, embedding_txt, logits_img, logits_txt

    def save_pretrained(self, save_directory, epoch=-1):
        """Save the model and its configuration to a directory.

        Args:
        - save_directory (str): Directory path to save the model.
        - epoch (int): Epoch number for saving different versions of the model (default: -1).

        Returns:
        - str: Path to the saved model file.

        """
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        assert os.path.isdir(save_directory)

        # Only save the model itself if we are not using distributed training
        model_to_save = self.module if hasattr(self, 'module') else self

        # Save BERT configuration
        model_to_save.bert_config.save_pretrained(save_directory)

        # Save the model state dict
        if epoch == -1:
            output_model_file = os.path.join(save_directory, 'pytorch_model.bin')
        else:
            output_model_file = os.path.join(save_directory, f'pytorch_model_epoch{epoch}.bin')
        torch.save(model_to_save.state_dict(), output_model_file)

        return output_model_file
