# gchtid_cnn

Simple exercise on classify gastric cancer histology patches
(GCHTID / HMU-GC-HE-30K, 8 tissue classes) with PyTorch CNNs 

Three model families live in `models.py`:

- `default` - a plain `nn.Sequential` CNN
- `custom` - VGG-style blocks, pick `n_blocks`, `base_filters`, `kernel_size`, `dropout`
- `vgg16` - torchvision VGG16 with ImageNet weights, `trainable` = `none` / `last_layer` / `last_block` / `all`

Reports test accuracy, macro precision / recall / specificity,
training time and inference time, and appends a row to `results.csv`.

## Setup

```bash
mamba env create -f environment.yml
mamba activate gchtid
```

Written against Python 3.12, PyTorch 2.9.1 (CUDA 12.9), torchvision 0.25, one GPU.

## Get the data

```bash
kaggle datasets download -d orvile/gastric-cancer-histopathology-tissue-image-dataset -p ~/datasets/gchtid --unzip
```

https://www.kaggle.com/datasets/orvile/gastric-cancer-histopathology-tissue-image-dataset

## Make the split

```bash
python data.py --root ~/datasets/gchtid/HMU-GC-HE-30K/all_image
```

- Remove duplicates by md5-hashes 
- Stratified 70/15/15 train/val/test split
- Default seed: 1004

## Train

```bash
python train.py --config configs/*.yaml
```

Saves `runs/<config name>/best.pt` (picked on val accuracy) and a confusion
matrix PNG, then evaluates on test once and appends to `results.csv`.

12 architecture runs, then 36 hyperparameter runs on `default`, the best `custom` and the best `vgg16`:

```bash
./run_experiments.sh
```

## Data source and license

Lou et al., *A large histological images dataset of gastric cancer with tumour
microenvironment annotation for AI*, Sci Data 12, 138 (2025).
https://doi.org/10.1038/s41597-025-04489-9

The data is on Figshare (https://doi.org/10.6084/m9.figshare.25954813) under
CC BY 4.0, and the Kaggle mirror used above has the same license. The paper
text is CC BY-NC-ND 4.0, but that only covers the article. No images are
included in this repo.

## License

MIT, see `LICENSE`.
