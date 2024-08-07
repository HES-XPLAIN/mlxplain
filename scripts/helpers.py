import os

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from torchvision import transforms

from scripts.custom_dataset import CustomDataset


def train_transform():
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    return transform


def test_transform():
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )
    return transform


def get_dataloaders(batch_size=16):
    data = pd.read_csv("./data/sports.csv")
    data["image_path"] = "./data/" + data["filepaths"]
    lbl = LabelEncoder()
    data["labels_encoded"] = lbl.fit_transform(data["labels"])
    # this image path is corrupted
    data = data[~data["image_path"].str.endswith(".lnk")]
    df_train = data[data["data set"] == "train"].reset_index(drop=True)
    # df_valid = data[data["data set"] == "valid"].reset_index(drop=True)
    df_test = data[data["data set"] == "test"].reset_index(drop=True)
    train_dataset = CustomDataset(df=df_train, transform=train_transform())
    # valid_dataset = CustomDataset(df=df_valid, transform=train_transform())
    test_dataset = CustomDataset(df=df_test, transform=test_transform())
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    # val_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


def load_images_to_ndarray(folder):
    image_list = []

    # Walk through the directory and collect image file paths
    for subdir, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(".jpg") or file.lower().endswith(".jpeg"):
                file_path = os.path.join(subdir, file)
                try:
                    # Open the image
                    img = Image.open(file_path).convert(
                        "RGB"
                    )  # Convert to RGB if needed
                    # Convert the image to a NumPy array and append to the list
                    img_array = np.array(img)
                    image_list.append(img_array)
                except Exception as e:
                    print(f"Error loading image {file_path}: {e}")

    # Stack the list of arrays into a single ndarray
    data = (
        np.stack(image_list, axis=0) if image_list else np.array([])
    )  # Handle empty case
    return data
