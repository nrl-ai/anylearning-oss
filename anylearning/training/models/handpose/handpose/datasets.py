import torch
from torch.utils.data import Dataset

import json
from pathlib import Path

from handpose.utils import normalize_landmarks


class HandPoseDataset(Dataset):
    def __init__(self, annotation_path, normalize=False):
        """
        Args:
            annotation_path: folder of per-image landmark JSONs.
            normalize: express the landmarks relative to the hand rather than
                to the image. Off by default because a model trained one way
                cannot be used the other, and models trained before this
                existed carry no such setting -- their stored config decides,
                which is what keeps them working.
        """
        super().__init__()

        self.annotation_files = list(Path(annotation_path).rglob("*.json"))
        self.normalize = normalize

    def __len__(self):
        return len(self.annotation_files)

    def __getitem__(self, index):
        annotation_file = self.annotation_files[index]

        with open(annotation_file, "r") as f:
            data = json.load(f)

        landmarks = data["data"]["landmarks"]
        landmarks_list = []
        for i in range(21):
            point = landmarks[str(i)]
            landmarks_list.append([point["x"], point["y"], point["z"]])

        if self.normalize:
            landmarks_list = normalize_landmarks(landmarks_list)

        landmarks_tensor = torch.Tensor(landmarks_list).reshape(1, -1)
        labels_tensor = torch.Tensor([data["data"]["label"]]).long()

        return landmarks_tensor, labels_tensor
