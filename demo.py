import os
import cv2
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from sod_model import SODModel


IMAGE_SIZE = 128
MODEL_PATH = "outputs/models/best_sod_model.pth"
DEMO_OUTPUT_DIR = "outputs/demo_results"

device = "cuda" if torch.cuda.is_available() else "cpu"


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
    resized_image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

    normalized_image = resized_image.astype(np.float32) / 255.0
    normalized_image = np.transpose(normalized_image, (2, 0, 1))

    tensor = torch.tensor(normalized_image, dtype=torch.float32)
    tensor = tensor.unsqueeze(0)

    return resized_image, tensor


def create_overlay(original_image, binary_mask):
    image_float = original_image.astype(np.float32) / 255.0

    overlay = image_float.copy()

    # red channel is increased where the object is predicted
    overlay[:, :, 0] = np.maximum(overlay[:, :, 0], binary_mask)

    # slightly darken green and blue where mask exists so red is clearer
    overlay[:, :, 1] = overlay[:, :, 1] * (1 - 0.35 * binary_mask)
    overlay[:, :, 2] = overlay[:, :, 2] * (1 - 0.35 * binary_mask)

    return np.clip(overlay, 0, 1)


def show_and_save_result(original_image, prediction, inference_time, save_path=None):
    prediction_np = prediction.squeeze().detach().cpu().numpy()

    # small smoothing only for nicer visualization
    prediction_np = cv2.GaussianBlur(prediction_np, (5, 5), 0)

    binary_mask = (prediction_np > 0.5).astype(np.float32)
    overlay = create_overlay(original_image, binary_mask)

    plt.figure(figsize=(14, 4))

    plt.subplot(1, 4, 1)
    plt.title("Input Image")
    plt.imshow(original_image)
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.title("Soft Saliency Mask")
    plt.imshow(prediction_np, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.title("Binary Mask")
    plt.imshow(binary_mask, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.title(f"Overlay\nInference: {inference_time:.4f}s")
    plt.imshow(overlay)
    plt.axis("off")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Demo result saved to: {save_path}")

    plt.show()


def run_demo():
    os.makedirs(DEMO_OUTPUT_DIR, exist_ok=True)

    print(f"Using device: {device}")
    print("Loading trained model...")

    model = load_trained_model(MODEL_PATH)

    image_path = input("Enter image path: ").strip()

    original_image, input_tensor = preprocess_image(image_path)
    input_tensor = input_tensor.to(device)

    start_time = time.time()

    with torch.no_grad():
        prediction = model(input_tensor)

    end_time = time.time()
    inference_time = end_time - start_time

    file_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(DEMO_OUTPUT_DIR, f"{file_name}_demo_result.png")

    show_and_save_result(
        original_image=original_image,
        prediction=prediction,
        inference_time=inference_time,
        save_path=save_path
    )


if __name__ == "__main__":
    run_demo()