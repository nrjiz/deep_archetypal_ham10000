# Deep Archetypal Analysis on HAM10000

This folder is a drop-in project that adapts your JAFFE Deep Archetypal Analysis (DAA) script to the **HAM10000** dataset.

Main goals:
- **One input argument**: `--data-dir /path/to/HAM10000_root`
- Optional **filters**: run on **all images** or only a subset (e.g. a single `dx` class or a single `localization`)
- Keep the **DAA objective + pipeline** structure (VAE-style reconstruction + side-info prediction + archetype loss)

## Expected HAM10000 folder layout

Your `--data-dir` should contain:

- `HAM10000_metadata.csv` (or similar name; auto-detected)
- `HAM10000_images_part_1/` (jpgs)
- `HAM10000_images_part_2/` (jpgs)

The script will scan both image folders and match paths using `image_id` from the metadata.

## Quick start

### 1) Train on the full dataset
```bash
python train_daa_ham10000.py \
  --data-dir /path/to/HAM10000 \
  --gpu 0 \
  --img-size 64 \
  --dim-latentspace 2 \
  --num_labels 7
```

### 2) Train within a single dx class (e.g. melanoma only)
```bash
python train_daa_ham10000.py \
  --data-dir /path/to/HAM10000 \
  --dx mel \
  --gpu 0 \
  --img-size 64
```

### 3) Train within a single localization (e.g. scalp only)
```bash
python train_daa_ham10000.py \
  --data-dir /path/to/HAM10000 \
  --localization scalp \
  --gpu 0 \
  --img-size 64
```

### 4) Resume / inference mode (loads last model)
```bash
python train_daa_ham10000.py \
  --data-dir /path/to/HAM10000 \
  --test-model \
  --results-path ./Results/HAM10000
```

### 5) Optional interpolation between two images (by image_id)
```bash
python train_daa_ham10000.py \
  --data-dir /path/to/HAM10000 \
  --test-model \
  --interp-from ISIC_0027419 \
  --interp-to   ISIC_0025030
```

## Notes

- Default split is a **group split by lesion_id** (recommended to avoid leakage).
- Image loading is **streaming** via `tf.data` (no need to load 10k images into RAM).
- Default encoder is `--encoder-arch convs` (closer to the conv encoder described in the paper).
- `--img-size 128` is supported, but will increase compute and VRAM use.

## Suggested conda env (example)

This code uses TF1-style graphs. The easiest setup is **TensorFlow 1.15** (or TF2 + compat.v1 mode).

Example:
```bash
conda create -n daa_ham python=3.8
conda activate daa_ham
pip install tensorflow==1.15.5 tensorflow-probability==0.8.0
pip install numpy pandas matplotlib scipy scikit-image imageio seaborn
```
