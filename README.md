# fp
> install kernelspec python3 to be used in jupyter notebook
```
poetry run python -m ipykernel install --user --display-name "eda"
```
> train the CNN
```
python im2text_matching.py --lr 1e-4 --cv_dir {path} --batch_size 128 --train_csv {path} --val_csv {path}
```
* If initialize weights by pre-training on ImageNet, training step takes only 2 epochs.
* If inialize weights randomly, training step takes 15 epochs.
After 2 epochs, cosine loss goes from ~1 to ~0.35. Save checkpoints. Perform transfer learning.