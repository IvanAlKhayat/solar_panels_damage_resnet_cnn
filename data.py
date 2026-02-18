from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    """
    Dataset class for solar cell defect classification.
    Handles loading, preprocessing, and augmentation of solar cell images.
    """

    def __init__(self, data, mode):
        """
        Initialize the dataset.

        Args:
            data: pandas.DataFrame containing image paths and labels from data.csv
            mode: str, either 'train' or 'val' to determine which transformations to apply
        """
        self.data = data
        self.mode = mode

        # Define transformations based on mode
        if mode == 'train':
            # Training: include data augmentation
            self.transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.RandomHorizontalFlip(p=0.5),
                tv.transforms.RandomVerticalFlip(p=0.5),
                tv.transforms.RandomRotation(degrees=15),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=train_mean, std=train_std)
            ])
        elif mode == 'val':
            # Validation: only basic transformations, no augmentation
            self.transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=train_mean, std=train_std)
            ])
        else:
            raise ValueError(f"Mode must be 'train' or 'val', got '{mode}'")

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: Number of samples in the dataset
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Get a sample from the dataset.

        Args:
            index: int, index of the sample to retrieve

        Returns:
            tuple: (image, label) where
                - image is a torch.Tensor of shape (3, H, W)
                - label is a torch.Tensor of shape (2,) containing [crack, inactive]
        """
        # Get the row from the dataframe
        row = self.data.iloc[index]

        # Load the image
        img_path = row['filename']
        image = imread(img_path)

        # Convert grayscale to RGB
        image = gray2rgb(image)

        # Apply transformations
        image = self.transform(image)

        # Get labels (crack and inactive)
        crack_label = row['crack']
        inactive_label = row['inactive']

        # Create label tensor
        label = torch.tensor([crack_label, inactive_label], dtype=torch.float32)

        return image, label