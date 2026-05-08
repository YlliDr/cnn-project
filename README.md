# Salient Object Detection using CNN

This project implements a Salient Object Detection (SOD) system from scratch using PyTorch.

The goal is to detect the most visually important object or region in an image and generate a saliency mask. The model takes an RGB image as input and outputs a one-channel mask where the salient object is highlighted.

---

## Project Overview

Salient Object Detection is a computer vision task that focuses on identifying the region in an image that naturally attracts the most attention.

This project includes a complete deep learning pipeline:

- Dataset loading and preprocessing
- Image and mask resizing
- Data augmentation
- CNN encoder-decoder model
- Custom BCE + IoU loss function
- Training and validation loop
- Model checkpoint saving
- Evaluation using IoU, Precision, Recall, F1 Score, and MAE
- Prediction visualizations
- Demo with inference time

---

## Dataset

The project uses the ECSSD dataset, which contains natural images and corresponding pixel-level saliency masks.

The dataset contains approximately:

- 1,000 RGB images
- 1,000 saliency masks

Dataset structure:

```text
data/
├── images/
│   ├── 0001.jpg
│   ├── 0002.jpg
│   └── ...
│
└── masks/
    ├── 0001.png
    ├── 0002.png
    └── ...
```

The dataset is not included in the repository because of size limitations.

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
├── .gitignore
└── README.md
```

---

## Main Files

```text
config.py          Stores project paths, image size, and training settings
data_loader.py     Loads images and masks, applies preprocessing and augmentation
sod_model.py       Contains the CNN encoder-decoder model
loss.py            Defines the BCE + IoU loss function
metrics.py         Contains evaluation metric functions
train.py           Trains and validates the model
evaluate.py        Evaluates the model and saves prediction examples
demo.py            Runs inference on a single image
requirements.txt   Lists required Python libraries
```

---

## Preprocessing

All images and masks are resized to 128 × 128 pixels.

Pixel values are normalized to the range [0, 1].

Masks are loaded as grayscale images.

Images and masks are converted to PyTorch tensors.

Dataset split:

```text
Training:   70%
Validation: 15%
Testing:    15%
```

---

## Data Augmentation

Data augmentation is applied only to the training set.

The used augmentations include:

```text
Horizontal flipping
Random cropping
Brightness adjustment
```

These techniques help reduce overfitting and improve generalization.

---

## Model Architecture

The model is a CNN encoder-decoder architecture built from scratch using PyTorch.

General flow:

```text
RGB Image → Encoder → Bottleneck → Decoder → Saliency Mask
```

The encoder extracts visual features from the input image, while the decoder reconstructs the saliency mask.

The model uses:

```text
Convolutional layers
ReLU activation
Batch Normalization
MaxPooling
Dropout
Upsampling layers
Skip connections
Final Sigmoid output
```

The final output is a one-channel mask with values between 0 and 1.

Values closer to 1 represent salient object pixels, while values closer to 0 represent background pixels.

---

## Loss Function

The model is trained using a hybrid loss function:

```text
Loss = BCE + 0.5 × (1 - IoU)
```

This combines Binary Cross Entropy for pixel-level classification and IoU loss for better mask overlap.

---

## Evaluation Metrics

The model is evaluated using:

```text
IoU        Measures overlap between predicted and ground-truth masks
Precision  Measures how many predicted salient pixels were correct
Recall     Measures how many actual salient pixels were detected
F1 Score   Balances Precision and Recall
MAE        Measures average pixel-level error
```

---

## Final Results

The best final result was achieved using:

```text
Threshold: 0.57
Test-Time Augmentation: Yes
Post-processing: Yes
```

Final test results:

```text
IoU:        0.6022
Precision: 0.7311
Recall:    0.8028
F1 Score:  0.7331
MAE:       0.2070
```

These results show that the model can detect most salient object regions with a good balance between Precision and Recall.

---

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## How to Train

Run:

```bash
python train.py
```

The training script saves model weights, checkpoints, logs, and plots inside the outputs folder.

---

## How to Evaluate

Run:

```bash
python evaluate.py
```

The evaluation script calculates the final metrics and saves prediction visualizations inside:

```text
outputs/predictions/
```

---

## How to Run Demo

Run:

```bash
python demo.py
```

A file selection window will open. Select an image from your computer.


The demo shows:

```text
Input image
Soft saliency mask
Binary mask
Overlay visualization
Inference time
```

Demo results are saved inside:

```text
outputs/demo_results/
```

---

## Notes

The following folders and files are ignored because they can be large or machine-specific:

```text
data/
outputs/
venv/
.venv/
__pycache__/
*.pth
*.pt
*.ckpt
```

The dataset and trained model files should be kept locally.

---

## Technologies Used

```text
Python
PyTorch
OpenCV
NumPy
Matplotlib
scikit-learn
tqdm
Visual Studio Code
Git / GitHub
```

---

## Conclusion

This project implements a complete Salient Object Detection pipeline using a CNN model built from scratch in PyTorch.

The model was trained and evaluated on the ECSSD dataset. The final configuration achieved an F1 Score of 0.7331 and an IoU of 0.6022, showing that the model can detect salient object regions with acceptable accuracy.

The project includes dataset loading, preprocessing, augmentation, model training, evaluation, prediction visualization, and demo inference.