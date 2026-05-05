# Salient Object Detection using CNN

This project implements a Salient Object Detection (SOD) model from scratch using PyTorch.  
The goal is to detect and highlight the most visually important object in an image by predicting a saliency mask.

---

## What it does

- Takes an input image
- Predicts a saliency mask (important regions)
- Shows overlay of prediction on the original image

---

## Project Structure

cnn-project/
│
├── data/
│   ├── images/
│   └── masks/
│
├── outputs/
│   ├── models/
│   └── predictions/
│
├── data_loader.py
├── sod_model.py
├── loss.py
├── metrics.py
├── train.py
├── evaluate.py
├── demo.py
├── requirements.txt
└── README.md

---

## Dataset

Dataset is NOT included due to size.

Use any Salient Object Detection dataset:
- ECSSD
- MSRA10K
- DUTS

After downloading, structure it like this:

data/
├── images/
└── masks/

Make sure images and masks match in order.

---

## Installation

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

---

## Training

python train.py

Model will be saved at:

outputs/models/best_sod_model.pth

---

## Evaluation

python evaluate.py

Metrics used:
- IoU
- Precision
- Recall
- F1 Score
- MAE

Results are saved in:

outputs/predictions/

---

## Demo

python demo.py

Enter image path when prompted.

The demo shows:
- Input image
- Predicted mask
- Overlay
- Inference time

---

## Model

Simple CNN encoder-decoder:
- Conv + ReLU + Pool (encoder)
- Upsampling layers (decoder)
- Sigmoid output

---

## Loss

Binary Cross Entropy + IoU

---

## Notes

Ignored in Git:
- data/
- outputs/
- venv/

---

## Author

Built as part of a deep learning project.