import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

from tqdm import tqdm
from torch.utils.data import DataLoader, random_split

from data_loader import SODDataset
from sod_model import SODModel
from config import *


# -------------------------------------------------
# Setup
# -------------------------------------------------
os.makedirs(PREDICTION_DIR, exist_ok=True)

RESULTS_PATH = os.path.join(PREDICTION_DIR, "test_results.txt")
BEST_RESULTS_PATH = os.path.join(PREDICTION_DIR, "best_threshold_results.txt")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# -------------------------------------------------
# Thresholds to test
# -------------------------------------------------
THRESHOLDS = [0.50, 0.52, 0.54, 0.55, 0.56, 0.57, 0.58, 0.60]


# -------------------------------------------------
# Load dataset without augmentation
# -------------------------------------------------
dataset = SODDataset(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR,
    image_size=IMAGE_SIZE,
    augment=False
)


# -------------------------------------------------
# Same split logic as training
# -------------------------------------------------
total_size = len(dataset)
train_size = int(0.70 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

_, _, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(SEED)
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# -------------------------------------------------
# Load trained model
# -------------------------------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. Train the model first using: python train.py"
    )

model = SODModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


# -------------------------------------------------
# Post-processing
# -------------------------------------------------
def postprocess_mask(mask):
    """
    Cleans binary predicted mask.
    Removes tiny white noise and fills tiny holes.
    """
    mask = (mask * 255).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)

    # remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # fill small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    mask = mask / 255.0
    return mask.astype(np.float32)


# -------------------------------------------------
# Metrics
# -------------------------------------------------
def calculate_binary_metrics(pred_binary, mask_binary, pred_raw, mask_raw):
    pred_binary = pred_binary.astype(np.float32)
    mask_binary = mask_binary.astype(np.float32)

    intersection = (pred_binary * mask_binary).sum()
    union = pred_binary.sum() + mask_binary.sum() - intersection

    iou = (intersection + 1e-6) / (union + 1e-6)

    tp = ((pred_binary == 1) & (mask_binary == 1)).sum()
    fp = ((pred_binary == 1) & (mask_binary == 0)).sum()
    fn = ((pred_binary == 0) & (mask_binary == 1)).sum()

    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    f1 = (2 * precision * recall) / (precision + recall + 1e-6)

    # MAE should use raw prediction, not binary prediction
    mae = np.mean(np.abs(pred_raw - mask_raw))

    return iou, precision, recall, f1, mae


# -------------------------------------------------
# Test-Time Augmentation prediction
# -------------------------------------------------
def predict_with_tta(images):
    """
    Predict normally and also predict horizontally flipped image.
    Then flip back and average both predictions.
    """
    predictions_normal = model(images)

    images_flipped = torch.flip(images, dims=[3])
    predictions_flipped = model(images_flipped)
    predictions_flipped = torch.flip(predictions_flipped, dims=[3])

    predictions = (predictions_normal + predictions_flipped) / 2

    return predictions


# -------------------------------------------------
# Evaluate one threshold
# -------------------------------------------------
def evaluate_threshold(threshold, use_postprocess=False):
    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_mae = 0.0
    sample_count = 0

    with torch.no_grad():
        for images, masks in tqdm(
            test_loader,
            desc=f"Testing threshold {threshold} | postprocess={use_postprocess}"
        ):
            images = images.to(device)
            masks = masks.to(device)

            predictions = predict_with_tta(images)

            predictions_np = predictions.cpu().numpy()
            masks_np = masks.cpu().numpy()

            for pred_np, mask_np in zip(predictions_np, masks_np):
                pred_np = pred_np.squeeze()
                mask_np = mask_np.squeeze()

                pred_binary = (pred_np > threshold).astype(np.float32)
                mask_binary = (mask_np > 0.5).astype(np.float32)

                if use_postprocess:
                    pred_binary = postprocess_mask(pred_binary)

                iou, precision, recall, f1, mae = calculate_binary_metrics(
                    pred_binary,
                    mask_binary,
                    pred_np,
                    mask_np
                )

                total_iou += iou
                total_precision += precision
                total_recall += recall
                total_f1 += f1
                total_mae += mae
                sample_count += 1

    results = {
        "threshold": threshold,
        "postprocess": use_postprocess,
        "iou": total_iou / sample_count,
        "precision": total_precision / sample_count,
        "recall": total_recall / sample_count,
        "f1": total_f1 / sample_count,
        "mae": total_mae / sample_count
    }

    return results


# -------------------------------------------------
# Save visualizations
# -------------------------------------------------
def save_visualization(image, mask, prediction, save_path, threshold, use_postprocess=False):
    image_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    mask_np = mask.squeeze().cpu().numpy()
    pred_np = prediction.squeeze().detach().cpu().numpy()

    pred_binary = (pred_np > threshold).astype(np.float32)

    if use_postprocess:
        pred_binary = postprocess_mask(pred_binary)

    overlay = image_np.copy()
    overlay[:, :, 0] = np.maximum(overlay[:, :, 0], pred_binary)

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
    plt.title(f"Prediction > {threshold}")
    plt.imshow(pred_binary, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.title("Overlay")
    plt.imshow(overlay)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# -------------------------------------------------
# Save final visual examples using best settings
# -------------------------------------------------
def save_best_visual_examples(best_threshold, best_postprocess):
    print("\nSaving visual examples using best settings...")

    with torch.no_grad():
        for index, (images, masks) in enumerate(tqdm(test_loader, desc="Saving examples")):
            if index >= 10:
                break

            images = images.to(device)
            masks = masks.to(device)

            predictions = predict_with_tta(images)

            save_path = os.path.join(
                PREDICTION_DIR,
                f"best_sample_{index + 1}_th_{best_threshold}_post_{best_postprocess}.png"
            )

            save_visualization(
                images[0].cpu(),
                masks[0].cpu(),
                predictions[0].cpu(),
                save_path,
                best_threshold,
                best_postprocess
            )


# -------------------------------------------------
# Main evaluation
# -------------------------------------------------
all_results = []

print("\nStarting automatic threshold search...")
print("This tests TTA with and without post-processing.\n")

for threshold in THRESHOLDS:
    # TTA only
    result_no_post = evaluate_threshold(threshold, use_postprocess=False)
    all_results.append(result_no_post)

    print(
        f"Threshold: {threshold:.2f} | "
        f"Postprocess: False | "
        f"IoU: {result_no_post['iou']:.4f} | "
        f"Precision: {result_no_post['precision']:.4f} | "
        f"Recall: {result_no_post['recall']:.4f} | "
        f"F1: {result_no_post['f1']:.4f} | "
        f"MAE: {result_no_post['mae']:.4f}"
    )

    # TTA + post-processing
    result_post = evaluate_threshold(threshold, use_postprocess=True)
    all_results.append(result_post)

    print(
        f"Threshold: {threshold:.2f} | "
        f"Postprocess: True  | "
        f"IoU: {result_post['iou']:.4f} | "
        f"Precision: {result_post['precision']:.4f} | "
        f"Recall: {result_post['recall']:.4f} | "
        f"F1: {result_post['f1']:.4f} | "
        f"MAE: {result_post['mae']:.4f}"
    )

    print("-" * 100)


# -------------------------------------------------
# Pick best result
# Priority:
# 1. Highest F1
# 2. If tied, highest IoU
# 3. If tied, lowest MAE
# -------------------------------------------------
best_result = sorted(
    all_results,
    key=lambda x: (x["f1"], x["iou"], -x["mae"]),
    reverse=True
)[0]


# -------------------------------------------------
# Prepare results text
# -------------------------------------------------
results_text = "\nAll Threshold Results\n"
results_text += "=" * 80 + "\n"

for result in all_results:
    results_text += (
        f"Threshold: {result['threshold']:.2f} | "
        f"Postprocess: {result['postprocess']} | "
        f"IoU: {result['iou']:.4f} | "
        f"Precision: {result['precision']:.4f} | "
        f"Recall: {result['recall']:.4f} | "
        f"F1 Score: {result['f1']:.4f} | "
        f"MAE: {result['mae']:.4f}\n"
    )

best_text = f"""
Best Final Test Results
Threshold: {best_result['threshold']}
Post-processing: {best_result['postprocess']}
IoU: {best_result['iou']:.4f}
Precision: {best_result['precision']:.4f}
Recall: {best_result['recall']:.4f}
F1 Score: {best_result['f1']:.4f}
MAE: {best_result['mae']:.4f}
"""

print(best_text)

results_text += "\n" + best_text


# -------------------------------------------------
# Save results
# -------------------------------------------------
with open(RESULTS_PATH, "w") as file:
    file.write(results_text)

with open(BEST_RESULTS_PATH, "w") as file:
    file.write(best_text)

print(f"Saved full results to: {RESULTS_PATH}")
print(f"Saved best result to: {BEST_RESULTS_PATH}")


# -------------------------------------------------
# Save visuals for best configuration
# -------------------------------------------------
save_best_visual_examples(
    best_threshold=best_result["threshold"],
    best_postprocess=best_result["postprocess"]
)

print(f"Saved visual examples in: {PREDICTION_DIR}")