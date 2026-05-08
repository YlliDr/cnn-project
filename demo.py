import os
import cv2
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

from sod_model import SODModel
from config import *


# -----------------------------
# Final demo settings
# -----------------------------
THRESHOLD = 0.57
USE_TTA = True
USE_POSTPROCESSING = True

device = "cuda" if torch.cuda.is_available() else "cpu"


def select_image_file():
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    file_path = filedialog.askopenfilename(
        title="Select an image for Salient Object Detection",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("All files", "*.*")
        ]
    )

    root.destroy()

    if not file_path:
        raise FileNotFoundError("No image was selected.")

    return file_path


def load_trained_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Train the model first by running: python train.py"
        )

    model = SODModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model


def preprocess_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path does not exist: {image_path}")

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Could not read the image. Check if the file is valid.")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    original_image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_LINEAR
    )

    normalized_image = original_image.astype(np.float32) / 255.0
    normalized_image = np.transpose(normalized_image, (2, 0, 1))

    tensor = torch.tensor(normalized_image, dtype=torch.float32).unsqueeze(0)

    return original_image, tensor


def postprocess_mask(mask):
    mask = (mask * 255).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)

    # Removes tiny white noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Fills tiny holes inside the object
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return (mask / 255.0).astype(np.float32)


def predict(model, input_tensor):
    if device == "cuda":
        torch.cuda.synchronize()

    start_time = time.time()

    with torch.no_grad():
        if USE_TTA:
            prediction_normal = model(input_tensor)

            flipped_input = torch.flip(input_tensor, dims=[3])
            prediction_flipped = model(flipped_input)
            prediction_flipped = torch.flip(prediction_flipped, dims=[3])

            prediction = (prediction_normal + prediction_flipped) / 2
        else:
            prediction = model(input_tensor)

    if device == "cuda":
        torch.cuda.synchronize()

    end_time = time.time()
    inference_time = end_time - start_time

    return prediction, inference_time


def create_overlay(original_image, binary_mask):
    image_float = original_image.astype(np.float32) / 255.0

    overlay = image_float.copy()

    # Red highlight for salient region
    overlay[:, :, 0] = np.maximum(overlay[:, :, 0], binary_mask)
    overlay[:, :, 1] = overlay[:, :, 1] * (1 - 0.35 * binary_mask)
    overlay[:, :, 2] = overlay[:, :, 2] * (1 - 0.35 * binary_mask)

    return np.clip(overlay, 0, 1)


def show_and_save_result(original_image, prediction, inference_time, image_path):
    prediction_np = prediction.squeeze().detach().cpu().numpy()

    soft_mask = cv2.GaussianBlur(prediction_np, (5, 5), 0)

    binary_mask = (soft_mask > THRESHOLD).astype(np.float32)

    if USE_POSTPROCESSING:
        binary_mask = postprocess_mask(binary_mask)

    overlay = create_overlay(original_image, binary_mask)

    os.makedirs(DEMO_OUTPUT_DIR, exist_ok=True)

    file_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(DEMO_OUTPUT_DIR, f"{file_name}_demo_result.png")

    plt.figure(figsize=(15, 5))
    plt.suptitle(
        f"Salient Object Detection Demo | Threshold: {THRESHOLD} | "
        f"TTA: {USE_TTA} | Post-processing: {USE_POSTPROCESSING} | "
        f"Inference: {inference_time:.4f}s",
        fontsize=11
    )

    plt.subplot(1, 4, 1)
    plt.title("Input Image")
    plt.imshow(original_image)
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.title("Soft Saliency Mask")
    plt.imshow(soft_mask, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.title("Binary Mask")
    plt.imshow(binary_mask, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.title("Overlay")
    plt.imshow(overlay)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    print("\nDemo completed successfully.")
    print(f"Selected image: {image_path}")
    print(f"Saved result: {save_path}")
    print(f"Inference time: {inference_time:.4f} seconds")


def run_demo():
    print("=" * 60)
    print("Salient Object Detection Demo")
    print("=" * 60)
    print(f"Using device: {device}")
    print(f"Threshold: {THRESHOLD}")
    print(f"Test-Time Augmentation: {USE_TTA}")
    print(f"Post-processing: {USE_POSTPROCESSING}")
    print("=" * 60)

    print("\nLoading trained model...")
    model = load_trained_model(MODEL_PATH)

    print("Select an image from the file picker window...")
    image_path = select_image_file()

    original_image, input_tensor = preprocess_image(image_path)
    input_tensor = input_tensor.to(device)

    prediction, inference_time = predict(model, input_tensor)

    show_and_save_result(
        original_image=original_image,
        prediction=prediction,
        inference_time=inference_time,
        image_path=image_path
    )


if __name__ == "__main__":
    run_demo()