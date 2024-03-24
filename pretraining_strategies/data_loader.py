import pandas as pd
import numpy as np
import warnings
import gensim
from PIL import Image
from torch.utils.data import Dataset
from xray_utils import get_from_csv, read_corpus, report2vec

# Disable DecompressionBombWarning and ignore it
Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

class MIMIC(Dataset):
    """
    A PyTorch Dataset for MIMIC dataset.

    Args:
        split (str): "train", "test", or "split" indicating the dataset split.
        data_path (str): Path to the CSV file containing image paths and annotations.
        transform: PyTorch transforms for image transformations and tensor conversion.
        annotate: Function to fill in np.nan as in the original chexpert paper.
    """
    def __init__(self, split, data_path, transform, annotate):
        # Read the CSV file containing image paths and annotations
        data = pd.read_csv(data_path)
        # Filter entries based on the split
        self.entries = data[data.split == split]
        self.transform = transform
        self.annotate = annotate

    def __getitem__(self, index):
        # Load image tensor and apply transformations
        img_path = self.entries.iloc[index, 1]
        img_tensor = self.transform(Image.open(img_path).convert('RGB'))
        # Get annotations and apply annotation function
        annotation = self.entries.iloc[index, 5:].values
        label = self.annotate(annotation)
        return img_tensor, label

    def __len__(self):
        return len(self.entries)

class OpenI(Dataset):
    """
    A PyTorch Dataset for OpenI dataset.

    Args:
        split (str): "train", "test", or "split" indicating the dataset split.
        entries_path (str): Path to the CSV file containing image paths and text reports.
        transform: PyTorch transforms for image transformations and tensor conversion.
        doc2vec_file (str, optional): Path to the doc2vec model file. Required for non-train splits.
    """
    def __init__(self, split, entries_path, transform, doc2vec_file=None):
        if not (doc2vec_file or split == "train"):
            raise ValueError("doc2vec must be provided for non-train splits")
        self.transform = transform
        # Get data entries from CSV based on the split
        self.entries = get_from_csv(entries_path, split)
        reports = self.entries.iloc[:, 2]
        # Load or infer doc2vec vectors for text reports
        self.doc2vec = report2vec(reports, doc2vec_file) if doc2vec_file else report2vec(reports)

    def __getitem__(self, index):
        # Load image tensor and apply transformations
        img_path = self.entries.iloc[index, 1]
        img_tensor = self.transform(Image.open(img_path).convert('RGB'))
        # Get text report and infer doc2vec vector
        report = self.entries.iloc[index, 2]
        report_tokens = gensim.utils.simple_preprocess(report)
        return img_tensor, self.doc2vec.infer_vector(report_tokens)

    def __len__(self):
        return len(self.entries)
