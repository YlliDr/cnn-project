import os
import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset


class SODDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=128, augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.augment = augment

        self.images = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ])

        self.masks = sorted([
            f for f in os.listdir(mask_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ])

        if len(self.images) != len(self.masks):
            raise ValueError("Number of images and masks must be the same")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.masks[index])

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Could not load image: {img_path}")

        if mask is None:
            raise ValueError(f"Could not load mask: {mask_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR
        )

        mask = cv2.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST
        )

        if self.augment:
            image, mask = self.apply_augmentation(image, mask)

        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0

        mask = (mask > 0.5).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        mask = np.expand_dims(mask, axis=0)

        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        return image, mask

    def random_crop(self, image, mask, crop_scale=0.9):
        h, w = image.shape[:2]

        crop_h = int(h * crop_scale)
        crop_w = int(w * crop_scale)

        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)

        image = image[top:top + crop_h, left:left + crop_w]
        mask = mask[top:top + crop_h, left:left + crop_w]

        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR
        )

        mask = cv2.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST
        )

        return image, mask

    def apply_augmentation(self, image, mask):
        # horizontal flip
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

        # random crop
        if random.random() > 0.5:
            image, mask = self.random_crop(image, mask, crop_scale=0.9)

        # brightness change only affects image, not mask
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            image = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        return image, mask