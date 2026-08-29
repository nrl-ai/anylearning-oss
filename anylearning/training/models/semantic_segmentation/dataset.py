import json
import os
import random
import re
from typing import Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


def is_an_image(file_name):
    fname = file_name.lower()
    all_extentions = [".jpg", ".png", ".jpeg", ".bmp"]
    return any(fname.endswith(ext) for ext in all_extentions)


class SegmentationDataset(Dataset):
    def __init__(
        self, image_dir: str, class_name2id: dict, transform=None, augmentation=None
    ):
        self.image_dir = image_dir
        self.transform = transform
        # Spatial augmentation belongs here rather than in `transform`, because
        # only here are the image and its mask both in hand. Flipping through
        # the image transform alone moves the picture and leaves the labels
        # behind -- the model then learns from annotations that no longer
        # describe what it is looking at, and nothing about the run looks wrong.
        self.augmentation = augmentation or {}
        self.class_name2id = class_name2id
        # sorted(): os.listdir() returns entries in filesystem order, which varies
        # between machines and even between runs. That made sample index -> image
        # non-deterministic, so a "reproducible" training run with a fixed seed
        # still saw a different sample order on a different box.
        self.images = sorted(f for f in os.listdir(image_dir) if is_an_image(f))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = re.sub(r"\.\w+$", ".json", img_path)

        # Read and resize image
        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size

        # Create and resize mask
        mask = self._create_mask_from_txt(mask_path, (img_h, img_w))

        if self.transform:
            image = self.transform(image)
            # resize mask to match image shape
            mask = cv2.resize(
                mask, (image.shape[2], image.shape[1]), interpolation=cv2.INTER_NEAREST
            )
            image, mask = self._augment(image, mask)

        mask = torch.from_numpy(np.ascontiguousarray(mask))
        mask = mask.long()

        return image, mask

    def _augment(self, image, mask):
        """Apply the same random spatial change to the image and the mask.

        Nearest-neighbour for the mask throughout: it holds class ids, and any
        interpolation between class 3 and class 5 invents a class 4.
        """
        if self.augmentation.get("horizontal_flip") and random.random() < 0.5:
            image = torch.flip(image, dims=[2])
            mask = np.fliplr(mask)
        if self.augmentation.get("vertical_flip") and random.random() < 0.5:
            image = torch.flip(image, dims=[1])
            mask = np.flipud(mask)

        degrees = self.augmentation.get("rotation_degrees") or 0
        if degrees:
            angle = random.uniform(-degrees, degrees)
            image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
            mask_tensor = torch.from_numpy(np.ascontiguousarray(mask))[None]
            mask = TF.rotate(
                mask_tensor, angle, interpolation=TF.InterpolationMode.NEAREST
            )[0].numpy()

        return image, mask

    def _create_mask_from_txt(
        self, mask_path: str, image_shape: Tuple[int, int]
    ) -> np.ndarray:
        mask = np.zeros(image_shape, dtype=np.uint8)
        if not os.path.exists(mask_path):
            return mask

        with open(mask_path, "r") as f:
            data = json.load(f)

        for annotation in data:
            if "points" not in annotation:  # defect annotation
                continue
            seg_data = annotation["points"]
            polygon_points = np.array(seg_data, dtype=np.int32)
            class_name = annotation["categories"][0]
            class_id = self.class_name2id[class_name]
            mask = cv2.fillPoly(mask, [polygon_points], color=class_id)

        return mask
