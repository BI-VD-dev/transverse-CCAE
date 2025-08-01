import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from torchvision.transforms import ToTensor
import json

# Custom dataset class that loads data on-the-fly
class ImageDataset(Dataset):
    def __init__(self, dataframe, image_folder, means, stds, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)  # Reset index for safe indexing
        self.image_folder = image_folder
        self.transform = transform
        self.means = means
        self.stds = stds

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        # Construct image path
        filename_in = os.path.basename(row.iloc[0])#.replace(".tiff", ".png")
        filename_out = os.path.basename(row.iloc[1])#.replace(".tiff", ".png")
        image_path_in = os.path.join(self.image_folder, filename_in)
        image_path_out = os.path.join(self.image_folder, filename_out)

        # Define transformations (normalize images)
        transform = transforms.Compose([
            #transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize between -1 and 1
        ])

        # # Load image (grayscale)
        # image = Image.open(image_path).convert("L")
        # if self.transform:
        #     image = self.transform(image)
        # Load image using torchvision (faster than PIL)
        try:
            image_in = transforms.functional.to_tensor(Image.open(image_path_in).convert("I"))/65535.0  # Normalize to [0,1]
            image_out = transforms.functional.to_tensor(Image.open(image_path_out).convert("I"))/65535.0  # Normalize to [0,1]
            image_in = transform(image_in)  # Apply the same transformations
            image_out = transform(image_out)  # Apply the same transformations
        except Exception as e:
            print(f"Error loading image {image_path_in} or {image_path_out}: {e}")
            # If image loading fails, return a dummy tensor
            image_in = torch.zeros((1, 512, 512), dtype=torch.float32)
            image_out = torch.zeros((1, 512, 512), dtype=torch.float32)

        conditions = np.array(row.iloc[-4:].values, dtype=np.float32)
        conditions = (conditions - self.means) / self.stds  # Normalize inputs using pre-computed means and stds
        # Load input variables (last 6 columns)
        conditions = torch.tensor(conditions, dtype=torch.float32)

        return image_in, image_out, conditions

# Function to create PyTorch DataLoaders (memory-efficient)
def get_dataloaders(batch_size=16, num_workers=4, pin_memory=False, csv_file=None, image_folder=None):
    # Load CSV file
    df_all = pd.read_csv(csv_file)

    # Compute normalization stats
    means = df_all.iloc[:, -4:].mean().values
    stds = df_all.iloc[:, -4:].std().values

    # Save normalization stats for later use
    with open("../VD-FEBE-Data/PreIP_512_FEB25/normalization.json", "w") as f:
        json.dump({"means": means.tolist(), "stds": stds.tolist()}, f)

    # Train-validation split (80/20)
    mask = np.random.rand(len(df_all)) < 0.8
    df_train = df_all[mask]
    df_val = df_all[~mask]

    # Create datasets (loads images on-the-fly)
    train_dataset = ImageDataset(df_train, image_folder, means, stds)
    val_dataset = ImageDataset(df_val, image_folder, means, stds)

    # Create DataLoaders (no large dataset stored in memory)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader

def preprocess_image(image_path):
# Define transformations (normalize images)
    transform = transforms.Compose([
        #transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize between -1 and 1
    ])

    image = transforms.functional.to_tensor(Image.open(image_path).convert("I"))/65535.0#Image.open(image_path).convert("I")  # Ensure grayscale
    image = transform(image)  # Apply the same transformations
    return image.unsqueeze(0)  # Add batch dimension for model input
