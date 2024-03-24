# Learning to Interpret Chest X-ray Images from Free-text Radiology Reports

### Exploratory Data Analysis (eda folder)
> Install the kernelspec to be used in jupyter notebook
```
poetry run python -m ipykernel install --user --display-name "eda"
```

### MI Maximisation (mi_maximisation folder)
> Set up the conda environment
```
conda env create -f conda_environment.yml
```
> Train the model
```
python training.py
```

### Pre-training Strategies (files in root folder)
> Train the CNN
```
python im2text_matching.py --lr 1e-4 --cv_dir {path} --batch_size 128 --train_csv {path} --val_csv {path}
```

| initialize weights | epochs taken by training step |
|:---:|:---:|
| by pre-training on ImageNet | 2 |
| randomly | 15 |
1. After 2 epochs, cosine loss goes from ~1 to ~0.35.
2. Save checkpoints. 
3. Perform transfer learning.
