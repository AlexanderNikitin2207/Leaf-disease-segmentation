# Leaf Disease Segmentation with SegFormer

Deep learning pipeline for segmenting plant disease lesions from RGB and multispectral imagery. The project evaluates convolutional and Transformer-based segmentation approaches on a public RGB dataset and transfers the selected model to real-world multispectral images of potato leaves affected by late blight.

## Overview

The workflow consists of two stages:

1. **RGB experiments** — compare fine-tuning strategies and segmentation architectures on a public leaf-disease dataset.
2. **Multispectral experiments** — adapt the selected SegFormer-B2 model to a real dataset containing visible- and infrared-spectrum images of potato plants.

The real-world dataset contains **604 high-resolution images (3456×5184)**. Images are split into 512×512 patches for training and evaluation.

## Model

Experiments on the public RGB dataset led to the selection of pretrained **SegFormer-B2** as the final architecture for multispectral fine-tuning.

## Results

Performance on the multispectral test set:

| Metric | Score |
| --- | ---: |
| Intersection over Union (IoU) | 0.413 |
| Dice coefficient | 0.584 |
| Cohen's kappa | 0.573 |

## Qualitative results

![Segmentation result 1](https://github.com/AlexanderNikitin2207/Leaf-disease-segmentation/assets/113451350/a8eb673a-938a-40d0-a789-cb61ecfcd6ac)

![Segmentation result 2](https://github.com/AlexanderNikitin2207/Leaf-disease-segmentation/assets/113451350/0a9da9dd-fbb6-49ed-974c-c7ad01e00c04)

## Data

The initial architecture experiments use the public [Leaf Disease Segmentation Dataset](https://www.kaggle.com/datasets/fakhrealam9537/leaf-disease-segmentation-dataset).

The final experiments use a real-world multispectral dataset of potato plants, including visible and infrared imagery. Segmentation masks are generated from polygon annotations describing affected regions.

### Multispectral examples

**Infrared image**

![Infrared image](https://github.com/AlexanderNikitin2207/Leaf-disease-segmentation/assets/113451350/4638d717-d2ea-4e8b-8567-552b8f0b5b29)

**Visible-spectrum image**

![Visible-spectrum image](https://github.com/AlexanderNikitin2207/Leaf-disease-segmentation/assets/113451350/ff39f90d-9200-416e-b679-2dbc92952055)

## Project structure

```text
.
├── rgb image experiments/
│   ├── prepare_rgb_dataset.ipynb
│   ├── fine_tuning_unet.ipynb
│   └── fine_tuning_transformers.ipynb
└── multispectral image experiments/
    ├── label_data.ipynb
    ├── augment_data.ipynb
    ├── Tune_HP.ipynb
    └── Segmormer_tuning_on_multispectral_images.ipynb
```

## Experiment pipeline

- Dataset preparation and train/validation/test split
- Comparison of U-Net-like and Transformer-based architectures
- Polygon-to-mask conversion for the multispectral dataset
- Data augmentation
- Hyperparameter tuning
- SegFormer-B2 fine-tuning
- Quantitative and qualitative segmentation evaluation

## Tech stack

Python · PyTorch · Torchvision · Hugging Face Transformers · SegFormer · segmentation-models-pytorch · Weights & Biases
