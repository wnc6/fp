import os
import logging
import csv
import sys
import numpy as np
from math import floor, ceil
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pandas as pd
import cv2
import random

import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

from .utils import MimicID

# Convert pulmonary edema severity (0-3) to one-hot encoding
def convert_to_onehot(severity):
    """Converts pulmonary edema severity to one-hot encoding.

    Args:
        severity (int): Severity of pulmonary edema (0-3).

    Returns:
        list: One-hot encoded representation of the severity.
        
    Raises:
        Exception: If severity value is not in the range [0, 3].
    """
    if severity == 0:
        return [1, 0, 0, 0]
    elif severity == 1:
        return [0, 1, 0, 0]
    elif severity == 2:
        return [0, 0, 1, 0]
    elif severity == 3:
        return [0, 0, 0, 1]
    else:
        raise Exception("No other possibilities of ordinal labels are possible")


class CXRImageReportDataset(torchvision.datasets.VisionDataset):
    """A CXR image-report dataset class that loads PNG images and 
    tokenized report text given a metadata file 
    and returns image, text batches. 

    Args:
        text_token_features (list): List of text token features.
        img_dir (string): Root directory for the CXR images.
        dataset_metadata (string): File path of the metadata 
            that will be used to construct this dataset. 
            This metadata file should contain data IDs that are used to
            load images and labels associated with data IDs.
        data_key (string): The name of the column that has image IDs.
        transform (callable, optional): A function/transform that takes in an image 
            and returns a transformed version.
        cache_images (bool, optional): Whether to cache images in memory.
    """
    
    def __init__(self, text_token_features, img_dir, dataset_metadata, 
                 data_key='mimic_id', transform=None, cache_images=False):
        super(CXRImageReportDataset, self).__init__(root=None, transform=transform)
        self.all_txt_tokens = {f.report_id: f.input_ids for f in text_token_features}
        self.all_txt_masks = {f.report_id: f.input_mask for f in text_token_features}
        self.all_txt_segments = {f.report_id: f.segment_ids for f in text_token_features}

        self.dataset_metadata = pd.read_csv(dataset_metadata)
        self.dataset_metadata['study_id'] = \
            self.dataset_metadata.apply(lambda row: \
                MimicID.get_study_id(row.mimic_id), axis=1)

        self.img_dir = img_dir
        self.data_key = data_key
        self.transform = transform
        self.image_ids = self.dataset_metadata[data_key]
        self.cache_images = cache_images
        if self.cache_images:
            self.cache_img_set() 
        else:
            self.images = None

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id, study_id = self.dataset_metadata.loc[idx, \
            [self.data_key, 'study_id']]

        txt = self.all_txt_tokens[study_id]
        txt = torch.tensor(txt, dtype=torch.long)

        txt_masks = self.all_txt_masks[study_id]
        txt_masks = torch.tensor(txt_masks, dtype=torch.long)

        txt_segments = self.all_txt_segments[study_id]
        txt_segments = torch.tensor(txt_segments, dtype=torch.long)

        if self.cache_images:
            img = self.images[str(idx)]
        else:
            png_path = os.path.join(self.img_dir, f'{img_id}.png')
            img = cv2.imread(png_path, cv2.IMREAD_ANYDEPTH)

        if self.transform is not None:
            img = self.transform(img)

        img = np.expand_dims(img, axis=0)

        return img, txt, txt_masks, txt_segments, study_id, img_id

    def cache_img_set(self):
        for idx in range(self.__len__()):
            img_id = self.dataset_metadata.loc[idx, self.data_key]
            png_path = os.path.join(self.data_dir, f'{img_id}.png')
            img = cv2.imread(png_path, cv2.IMREAD_ANYDEPTH)
            if idx == 0:
                self.images = {}
            self.images[str(idx)] = img


# adapted from
# https://towardsdatascience.com/https-medium-com-chaturangarajapakshe-text-classification-with-transformer-models-d370944b50ca
def load_and_cache_examples(args, tokenizer):
    """Load and cache text examples.

    This function loads pre-processed text features if they exist, otherwise preprocesses the raw text data
    and saves the features.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        tokenizer: Tokenizer object for tokenizing text data.

    Returns:
        torch.Tensor: Cached text features.
    """
    logger = logging.getLogger(__name__)

    # Initialize data processor and get the number of labels
    processor = ClassificationDataProcessor()
    num_labels = len(processor.get_labels())

    # Define path for cached features file
    cached_features_file = os.path.join(args.text_data_dir, f"cachedfeatures_train_seqlen-{args.max_seq_length}")

    # Check if cached features file exists
    if os.path.exists(cached_features_file):
        logger.info(f"Loading features from cached file {cached_features_file}")
        print(f"Loading features from cached file {cached_features_file}")
        features = torch.load(cached_features_file)
    else:
        logger.info(f"Creating features from dataset file at {args.text_data_dir}")
        print(f"Creating features from dataset file at {args.text_data_dir}")
        label_list = processor.get_labels()
        examples = processor.get_all_examples(args.text_data_dir)
        features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer)
        logger.info(f"Saving features into cached file {cached_features_file}")
        print(f"Saving features into cached file {cached_features_file}")
        torch.save(features, cached_features_file)

    return features


class InputFeatures(object):
    """A single set of features of text data."""
    
    def __init__(self, input_ids, input_mask, segment_ids, label_id, report_id):
        """Constructs an InputFeatures object.
        
        Args:
            input_ids (torch.Tensor): Tensor containing input IDs.
            input_mask (torch.Tensor): Tensor containing input masks.
            segment_ids (torch.Tensor): Tensor containing segment IDs.
            label_id (int): Label ID.
            report_id (str): Report ID.
        """
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.report_id = report_id


class InputExample(object):
    """A single training/test example for simple sequence classification."""
    
    def __init__(self, report_id, guid, text_a, text_b=None, labels=None):
        """Constructs an InputExample object.
        
        Args:
            report_id (str): Report ID.
            guid: Unique ID for the example.
            text_a (str): The untokenized text of the first sequence.
            text_b (str, optional): The untokenized text of the second sequence (for sequence pair tasks).
            labels (list, optional): Labels of the example.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.labels = labels
        self.report_id = report_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    
    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab-separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

class ClassificationDataProcessor(DataProcessor):
    """Processor for multi-class classification dataset.
    Assume reading from a multiclass file
    so the label will be in 0-3 format.
    """

    def get_train_examples(self, data_dir):
        """Get examples from the training set."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """Get examples from the development set."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_all_examples(self, data_dir):
        """Get examples from the entire dataset."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "all_data.tsv")), "dev")

    def get_labels(self):
        """Get the list of possible labels."""
        return ["0", "1", "2", "3"]

    def _create_examples(self, lines, set_type):
        """Create examples for the dataset."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = f"{set_type}-{i}"
            text_a = line[-1]
            labels = line[1]
            report_id = line[2]
            examples.append(
                InputExample(
                    report_id=report_id, guid=guid,
                    text_a=text_a, text_b=None, labels=labels))
        return examples


def convert_example_to_feature(example_row):
    """Convert an example to a feature."""
    example, label_map, max_seq_length, tokenizer = example_row

    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    if tokens_b:
        tokens += tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = label_map[example.labels]
    return InputFeatures(input_ids=input_ids,
                         input_mask=input_mask,
                         segment_ids=segment_ids,
                         label_id=label_id, 
                         report_id=example.report_id)


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Convert a list of examples to features."""
    label_map = {label: i for i, label in enumerate(label_list)}
    label_map['-1'] = -1  # To handle '-1' label (i.e. unlabeled data)
    examples_for_processing = [(example, label_map, max_seq_length, tokenizer) \
        for example in examples]
    process_count = cpu_count() - 1
    with Pool(process_count) as p:
            features = list(tqdm(p.imap(convert_example_to_feature, 
                                 examples_for_processing), 
                            total=len(examples)))
    return features
