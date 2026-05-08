# Salient Object Detection using CNN

This project implements a Salient Object Detection (SOD) system from scratch using PyTorch.

The goal is to detect the most visually important object or region in an image and generate a saliency mask. The model takes an RGB image as input and outputs a one-channel mask showing the predicted salient area.

---

## Project Overview

Salient Object Detection is a computer vision task where the model learns to identify the part of an image that naturally attracts human attention.

This project includes a complete deep learning pipeline:

- Dataset loading and preprocessing
- Image and mask resizing
- Data augmentation
- CNN encoder-decoder model
- Custom loss function
- Training and validation loop
- Checkpoint saving and resume support
- Evaluation using multiple metrics
- Prediction visualizations
- Demo with inference time

---

## Project Structure

```text
cnn-project/
│
├── data/
│   ├── images/
│   └── masks/
│
├── outputs/
│   ├── models/
│   ├── predictions/
│   ├── plots/
│   ├── logs/
│   └── demo_results/
│
├── config.py
├── data_loader.py
├── sod_model.py
├── loss.py
├── metrics.py
├── train.py
├── evaluate.py
├── demo.py
├── requirements.txt
└── README.md