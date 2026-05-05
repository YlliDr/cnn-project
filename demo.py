import cv2
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from sod_model import SODModel


IMAGE_SIZE = 128
MODEL_PATH = "outputs/models/best_sod_model.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"


def preprocess_image(image_path):
    # read image from path
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Image not found. Please check the path.")

    # opencv reads images as BGR, so we convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # resize image to the same size used during training
    resized_image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

    # normalize pixel values
    normalized_image = resized_image.astype(np.float32) / 255.0

    # change shape from HWC to CHW because PyTorch expects this format
    normalized_image = np.transpose(normalized_image, (2, 0, 1))

    # convert image to tensor and add batch dimension
    tensor = torch.tensor(normalized_image, dtype=torch.float32)
    tensor = tensor.unsqueeze(0)

    return resized_image, tensor


def show_result(original_image, prediction, inference_time):
    # remove extra dimensions from prediction
    prediction_np = prediction.squeeze().detach().cpu().numpy()

    # create simple red overlay on the original image
    overlay = original_image.astype(np.float32) / 255.0
    overlay[:, :, 0] = np.maximum(overlay[:, :, 0], prediction_np)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.title("Input Image")
    plt.imshow(original_image)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Predicted Mask")
    plt.imshow(prediction_np, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title(f"Overlay\nTime: {inference_time:.4f}s")
    plt.imshow(overlay)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# load model
model = SODModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ask user for an image path
image_path = input("Enter image path: ")

original_image, input_tensor = preprocess_image(image_path)
input_tensor = input_tensor.to(device)

# measure inference time
start_time = time.time()

with torch.no_grad():
    prediction = model(input_tensor)

end_time = time.time()

inference_time = end_time - start_time

show_result(original_image, prediction, inference_time)