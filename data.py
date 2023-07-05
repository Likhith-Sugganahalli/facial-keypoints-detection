import os
import numpy as np
import pandas as pd
import imageio as imgio
import matplotlib.pyplot as plt

import torch
import torchvision

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms, utils


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, keypoints = sample["image"], sample["keypoints"]

        image = np.expand_dims(image, axis=-1)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {
            "image": torch.from_numpy(image),
            "keypoints": torch.from_numpy(keypoints),
        }


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, keypoints = sample["image"], sample["keypoints"]
        # print("keypoints", keypoints.shape)

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        y_origin = np.random.randint(0, h - new_h)
        x_origin = np.random.randint(0, w - new_w)

        image = image[y_origin : y_origin + new_h, x_origin : x_origin + new_w]

        keypoints = keypoints - [x_origin, y_origin]

        x_max_conditional = keypoints[:, 0] < new_w
        keypoints = keypoints[x_max_conditional]

        x_origin_conditional = keypoints[:, 0] > 0
        keypoints = keypoints[x_origin_conditional]

        y_max_conditional = keypoints[:, 1] < new_h
        keypoints = keypoints[y_max_conditional]

        y_origin_conditional = keypoints[:, 1] > 0
        keypoints = keypoints[y_origin_conditional]

        return {"image": image, "keypoints": keypoints}


def loadImage(image_col_list, shape_tuple=96):
    if isinstance(shape_tuple, int):
        shape_tuple = (shape_tuple, shape_tuple)

    img_list = image_col_list.split(" ")
    temp = np.asarray(img_list)
    temp.resize(shape_tuple)
    temp = temp.astype(np.uint8)

    return temp


def loadKeyPoints(keypoints_row):
    keypoints = []
    for i in range(0, len(keypoints_row), 2):
        keypoints.append([keypoints_row[i], keypoints_row[i + 1]])
    return np.array(keypoints)


class FaceKeypointsDataset(Dataset):
    """Face Keypoints dataset."""

    def __init__(self, csv_file, root_dir="./", transform: transforms.Compose = None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = loadImage(self.landmarks_frame.iloc[idx, -1])

        keypoints = loadKeyPoints(self.landmarks_frame.iloc[idx, :-1].to_list())

        sample = {"image": image, "keypoints": keypoints}

        if self.transform:
            # for transform in self.transform:
            sample = self.transform(sample)

        return sample


def main():
    composed = transforms.Compose([RandomCrop(90)])

    face_dataset = FaceKeypointsDataset(
        csv_file="./data/training.csv", transform=composed
    )

    for i in range(len(face_dataset)):
        sample = face_dataset[i]

        plt.imshow(sample["image"])
        plt.scatter(
            sample["keypoints"][:, 0],
            sample["keypoints"][:, 1],
            s=10,
            marker=".",
            c="r",
        )
        plt.show()

        if i == 10:
            break


if __name__ == "__main__":
    main()
