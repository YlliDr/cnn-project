import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split

from data_loader import SODDataset
from sod_model import SODModel
from loss import sod_loss
from metrics import calculate_metrics


# local dataset paths
IMAGE_DIR = "data/images"
MASK_DIR = "data/masks"

# basic training settings
IMAGE_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-3

# where we save model files
MODEL_DIR = "outputs/models"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_sod_model.pth")
LAST_CHECKPOINT_PATH = os.path.join(MODEL_DIR, "last_checkpoint.pth")

os.makedirs(MODEL_DIR, exist_ok=True)

# use gpu if available, otherwise cpu
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# load the full dataset
dataset = SODDataset(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR,
    image_size=IMAGE_SIZE,
    augment=True
)

# split dataset into train, validation, and test
total_size = len(dataset)
train_size = int(0.70 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# create model and optimizer
model = SODModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_val_loss = float("inf")
start_epoch = 0


# resume training if there is already a checkpoint
if os.path.exists(LAST_CHECKPOINT_PATH):
    checkpoint = torch.load(LAST_CHECKPOINT_PATH, map_location=device)

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint["best_val_loss"]

    print(f"Resumed training from epoch {start_epoch + 1}")


for epoch in range(start_epoch, EPOCHS):
    model.train()
    train_loss = 0.0

    train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} - Training")

    for images, masks in train_bar:
        images = images.to(device)
        masks = masks.to(device)

        # forward pass
        predictions = model(images)

        # calculate loss
        loss = sod_loss(predictions, masks)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        train_bar.set_postfix({
            "loss": loss.item()
        })

    train_loss = train_loss / len(train_loader)


    # validation after each epoch
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

    val_loss = val_loss / len(val_loader)
    val_iou = val_iou / len(val_loader)
    val_precision = val_precision / len(val_loader)
    val_recall = val_recall / len(val_loader)
    val_f1 = val_f1 / len(val_loader)
    val_mae = val_mae / len(val_loader)

    print()
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val IoU: {val_iou:.4f}")
    print(f"Val Precision: {val_precision:.4f}")
    print(f"Val Recall: {val_recall:.4f}")
    print(f"Val F1: {val_f1:.4f}")
    print(f"Val MAE: {val_mae:.4f}")
    print()

    # save checkpoint so training can continue later
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_loss": best_val_loss
    }

    torch.save(checkpoint, LAST_CHECKPOINT_PATH)
    print("Checkpoint saved.")

    # save best model based on validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print("Best model saved.")

print("Training finished.")