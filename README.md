# Learning to Interpret Chest X-ray Images from Free-text Radiology Reports

> Install kernelspec python3 to be used in jupyter notebook
```
poetry run python -m ipykernel install --user --display-name "eda"
```
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