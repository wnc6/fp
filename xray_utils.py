import pandas as pd
import numpy as np
import gensim
import torch
import torchvision.transforms as transforms
import torchvision.datasets as torchdata
import torchvision.models as torchmodels

def transform_xray():
    """
    Define the transformation pipeline for X-ray images.

    Returns:
        transform: torchvision.transforms.Compose, the transformation pipeline.
    """
    # Obtained by averaging/std'ing over all train images
    mean = [109.99]
    std = [53.95]
    transform = transforms.Compose([
        transforms.Resize((300,300)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return transform

def get_from_csv(entries_path, split):
    """
    Load data from a CSV file.

    Args:
        entries_path (str): Path to the directory containing the CSV files.
        split (str): The dataset split ("train", "test", "val").

    Returns:
        data: pandas.DataFrame, the loaded data.
    """
    data = pd.read_csv(f'{entries_path}/{split}_entries.csv')[['label','xray_paths','text']]
    # Adjusting labels to fit with Snorkel MeTaL labeling convention (0 reserved for abstain)
    data['label'][data['label']==0] = 2
    perc_pos = sum(data['label']==1)/len(data)
    print(f'{len(data)} {split} examples: {100*perc_pos:0.1f}% Abnormal')
        
    return data

def read_corpus(reports):
    """
    Convert reports into a format suitable for gensim's Doc2Vec.

    Args:
        reports (iterable): An iterable of reports.

    Yields:
        gensim.models.doc2vec.TaggedDocument, a tagged document.
    """
    for i, line in enumerate(reports):
        tokens = gensim.utils.simple_preprocess(line)
        yield gensim.models.doc2vec.TaggedDocument(tokens, [i])
        
def report2vec(reports, file=None, file_save="report2vec.model"):
    """
    Train a Doc2Vec model on the given reports or load an existing model.

    Args:
        reports (iterable): An iterable of reports.
        file (str, optional): Path to an existing Doc2Vec model file.
        file_save (str, optional): Path to save the trained Doc2Vec model.

    Returns:
        model: gensim.models.doc2vec.Doc2Vec, the trained or loaded Doc2Vec model.
    """
    if file:
        return gensim.utils.SaveLoad.load(file)
    
    corpus = list(read_corpus(reports))
    model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=1, epochs=40)
    model.build_vocab(corpus)
    model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
    model.save(file_save)
    return model

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

def get_model(embed_size):
    """
    Get a pre-trained DenseNet model for feature extraction.

    Args:
        embed_size (int): The size of the embedding layer.

    Returns:
        dnet: torch.nn.Module, the DenseNet model.
    """
    dnet = torchmodels.densenet161(pretrained=True)
    set_parameter_requires_grad(dnet, False)
    num_ftrs = dnet.classifier.in_features
    dnet.classifier = torch.nn.Linear(num_ftrs, embed_size)

    return dnet
