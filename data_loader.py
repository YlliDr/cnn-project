import os
import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset


class SODDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=128, augment=False):
        # paths to images and masks
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        # size we want to resize everything to
        self.image_size = image_size

        # whether we apply augmentation or not
        self.augment = augment

        # get all file names and keep them sorted so they match
        self.images = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ])

        self.masks = sorted([
            f for f in os.listdir(mask_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ])

        # quick sanity check
        if len(self.images) != len(self.masks):
            raise ValueError("Number of images and masks must be the same")

    def __len__(self):
        # total number of samples
        return len(self.images)

    def __getitem__(self, index):
        # build full paths
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.masks[index])

        # read image and mask
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Could not load image: {img_path}")

        if mask is None:
            raise ValueError(f"Could not load mask: {mask_path}")

        # convert BGR to RGB (opencv loads as BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # resize both
        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size))

        # apply simple augmentation if enabled
        if self.augment:
            image, mask = self.apply_augmentation(image, mask)

        # normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0

        # change shape from HWC -> CHW
        image = np.transpose(image, (2, 0, 1))

        # add channel dimension to mask
        mask = np.expand_dims(mask, axis=0)

        # convert to tensors
        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        return image, mask

    def apply_augmentation(self, image, mask):
        # horizontal flip
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

        # small brightness change
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            image = np.clip(image * factor, 0, 255).astype(np.uint8)

        return image, mask