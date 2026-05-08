import os
import csv
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import DataLoader, Subset, random_split

from data_loader import SODDataset
from sod_model import SODModel
from loss import sod_loss
from metrics import calculate_metrics
from config import *


# Derived paths

BEST_MODEL_PATH = MODEL_PATH
LAST_CHECKPOINT_PATH = CHECKPOINT_PATH
HISTORY_PATH = os.path.join(LOG_DIR, "training_history.csv")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# Reproducibility

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# Device

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# Dataset split

base_dataset = SODDataset(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR,
    image_size=IMAGE_SIZE,
    augment=False
)

total_size = len(base_dataset)
train_size = int(0.70 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

train_subset, val_subset, test_subset = random_split(
    base_dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(SEED)
)


# Separate datasets
# Augmentation ONLY for training

train_full_dataset = SODDataset(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR,
    image_size=IMAGE_SIZE,
    augment=True
)

val_full_dataset = SODDataset(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR,
    image_size=IMAGE_SIZE,
    augment=False
)

test_full_dataset = SODDataset(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR,
    image_size=IMAGE_SIZE,
    augment=False
)

train_dataset = Subset(train_full_dataset, train_subset.indices)
val_dataset = Subset(val_full_dataset, val_subset.indices)
test_dataset = Subset(test_full_dataset, test_subset.indices)

print(f"Total samples: {total_size}")
print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")


# DataLoaders

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# Model and optimizer

model = SODModel().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

best_val_loss = float("inf")
start_epoch = 0
epochs_without_improvement = 0
history = []


# Resume checkpoint if available

if os.path.exists(LAST_CHECKPOINT_PATH):
    checkpoint = torch.load(LAST_CHECKPOINT_PATH, map_location=device)

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint["best_val_loss"]
    epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)
    history = checkpoint.get("history", [])

    print(f"Resumed training from epoch {start_epoch + 1}")


# Training loop

for epoch in range(start_epoch, EPOCHS):
    model.train()
    train_loss = 0.0

    train_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS} - Training"
    )

    for images, masks in train_bar:
        images = images.to(device)
        masks = masks.to(device)

        predictions = model(images)
        loss = sod_loss(predictions, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        train_bar.set_postfix({
            "loss": f"{loss.item():.4f}"
        })

    avg_train_loss = train_loss / len(train_loader)


    # Validation

    model.eval()

    val_loss = 0.0
    val_iou = 0.0
    val_precision = 0.0
    val_recall = 0.0
    val_f1 = 0.0
    val_mae = 0.0

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Validation"):
            images = images.to(device)
            masks = masks.to(device)

            predictions = model(images)

            loss = sod_loss(predictions, masks)
            val_loss += loss.item()

            metrics = calculate_metrics(predictions, masks)

            val_iou += metrics["iou"]
            val_precision += metrics["precision"]
            val_recall += metrics["recall"]
            val_f1 += metrics["f1"]
            val_mae += metrics["mae"]

    avg_val_loss = val_loss / len(val_loader)
    avg_val_iou = val_iou / len(val_loader)
    avg_val_precision = val_precision / len(val_loader)
    avg_val_recall = val_recall / len(val_loader)
    avg_val_f1 = val_f1 / len(val_loader)
    avg_val_mae = val_mae / len(val_loader)


    # Print epoch results

    print()
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Val Loss: {avg_val_loss:.4f}")
    print(f"Val IoU: {avg_val_iou:.4f}")
    print(f"Val Precision: {avg_val_precision:.4f}")
    print(f"Val Recall: {avg_val_recall:.4f}")
    print(f"Val F1: {avg_val_f1:.4f}")
    print(f"Val MAE: {avg_val_mae:.4f}")
    print()


    # Save history CSV

    epoch_result = {
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "val_iou": avg_val_iou,
        "val_precision": avg_val_precision,
        "val_recall": avg_val_recall,
        "val_f1": avg_val_f1,
        "val_mae": avg_val_mae
    }

    history.append(epoch_result)

    with open(HISTORY_PATH, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=epoch_result.keys())
        writer.writeheader()
        writer.writerows(history)

    print(f"Training history saved to {HISTORY_PATH}")


    # Save best model

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0

        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print("Best model saved.")

    else:
        epochs_without_improvement += 1
        print(f"No improvement for {epochs_without_improvement} epoch(s).")


    # Save checkpoint

    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "epochs_without_improvement": epochs_without_improvement,
        "history": history
    }

    torch.save(checkpoint, LAST_CHECKPOINT_PATH)
    print("Checkpoint saved.")


    # Early stopping

    if epochs_without_improvement >= PATIENCE:
        print("Early stopping triggered.")
        break


# Save training plots

if len(history) > 0:
    epochs = [row["epoch"] for row in history]
    train_losses = [row["train_loss"] for row in history]
    val_losses = [row["val_loss"] for row in history]
    val_ious = [row["val_iou"] for row in history]
    val_f1s = [row["val_f1"] for row in history]

    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(PLOT_DIR, "loss_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(epochs, val_ious, label="Validation IoU")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.title("Validation IoU")
    plt.legend()
    plt.savefig(os.path.join(PLOT_DIR, "iou_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(epochs, val_f1s, label="Validation F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("Validation F1 Score")
    plt.legend()
    plt.savefig(os.path.join(PLOT_DIR, "f1_curve.png"))
    plt.close()

    print("Training plots saved.")

print("Training finished.")