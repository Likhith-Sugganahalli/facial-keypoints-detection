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

from data import FaceKeypointsDataset, ToTensor, RandomCrop


# Helper function to show a batch
def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, landmarks_batch = sample_batched["image"], sample_batched["keypoints"]
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.scatter(
            landmarks_batch[i, :, 0].numpy() + i * im_size + (i + 1) * grid_border_size,
            landmarks_batch[i, :, 1].numpy() + grid_border_size,
            s=10,
            marker=".",
            c="r",
        )

        plt.title("Batch from dataloader")


if __name__ == "__main__":
    composed = transforms.Compose([ToTensor()])  # RandomCrop(90)

    face_dataset = FaceKeypointsDataset(
        csv_file="./data/training.csv", transform=composed
    )

    # for i in range(len(face_dataset)):
    #     sample = face_dataset[i]

    #     print(i, sample["image"].size(), sample["keypoints"].size())

    #     if i == 3:
    #         break

    dataloader = DataLoader(face_dataset, batch_size=1, shuffle=False, num_workers=0)
    num_keypoints = []
    for i_batch, sample_batched in enumerate(dataloader):
        # print(
        #     i_batch, sample_batched["image"].size(), sample_batched["keypoints"].size()
        # )
        num_keypoints.append(sample_batched["keypoints"].size())
        # plt.figure()
        # show_landmarks_batch(sample_batched)
        # plt.axis("off")
        # plt.ioff()
        # # plt.show()
        # plt.savefig(f"./vis_images/{i_batch}.png")
        # plt.close()

    plt.hist(num_keypoints)
    # plt.plot(num_keypoints, len(num_keypoints) * [1], "x")
    plt.show()
