import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split

from data_loader import SODDataset
from sod_model import SODModel
from metrics import calculate_metrics


# local dataset paths
IMAGE_DIR = "data/images"
MASK_DIR = "data/masks"

# same image size used during training
IMAGE_SIZE = 128
BATCH_SIZE = 1

# saved model and output folder
MODEL_PATH = "outputs/models/best_sod_model.pth"
PREDICTION_DIR = "outputs/predictions"

os.makedirs(PREDICTION_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# load dataset without augmentation for testing
dataset = SODDataset(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR,
    image_size=IMAGE_SIZE,
    augment=False
)

# use the same split logic as training
total_size = len(dataset)
train_size = int(0.70 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# load trained model
model = SODModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


def save_visualization(image, mask, prediction, save_path):
    # convert tensors back to normal image format
    image_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    mask_np = mask.squeeze().cpu().numpy()
    pred_np = prediction.squeeze().detach().cpu().numpy()

    # red overlay shows where the model thinks the salient object is
    overlay = image_np.copy()
    overlay[:, :, 0] = np.maximum(overlay[:, :, 0], pred_np)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 4, 1)
    plt.title("Input")
    plt.imshow(image_np)
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.title("Ground Truth")
    plt.imshow(mask_np, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.title("Prediction")
    plt.imshow(pred_np, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.title("Overlay")
    plt.imshow(overlay)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# totals for final average metrics
total_iou = 0.0
total_precision = 0.0
total_recall = 0.0
total_f1 = 0.0
total_mae = 0.0
sample_count = 0


with torch.no_grad():
    for index, (images, masks) in enumerate(tqdm(test_loader, desc="Testing")):
        images = images.to(device)
        masks = masks.to(device)

        predictions = model(images)

        metrics = calculate_metrics(predictions, masks)

        total_iou += metrics["iou"]
        total_precision += metrics["precision"]
        total_recall += metrics["recall"]
        total_f1 += metrics["f1"]
        total_mae += metrics["mae"]

        sample_count += 1

        # save a few examples for the report
        if index < 10:
            save_path = os.path.join(PREDICTION_DIR, f"sample_{index + 1}.png")

            save_visualization(
                images.cpu(),
                masks.cpu(),
                predictions.cpu(),
                save_path
            )


print()
print("Final Test Results")
print(f"IoU: {total_iou / sample_count:.4f}")
print(f"Precision: {total_precision / sample_count:.4f}")
print(f"Recall: {total_recall / sample_count:.4f}")
print(f"F1 Score: {total_f1 / sample_count:.4f}")
print(f"MAE: {total_mae / sample_count:.4f}")
print()
print(f"Saved visual examples in: {PREDICTION_DIR}")