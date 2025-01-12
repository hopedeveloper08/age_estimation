import os

import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from PIL import Image

import torchvision.transforms as transforms


class UTKDataset(Dataset):

    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        path = os.path.join(self.root_dir, item['image_name'])
        image = Image.open(path)
        if self.transform is not None:
            image = self.transform(image)
        age = torch.tensor(item['age'], dtype=torch.float32)
        return image, age


def get_dataloaders():
    train_transformation = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transformation = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    root_dir = '../data/raw/'
    train_dataset = UTKDataset(root_dir, '../data/preprocess/train.csv', train_transformation)
    valid_dataset = UTKDataset(root_dir, '../data/preprocess/valid.csv', test_transformation)
    test_dataset = UTKDataset(root_dir, '../data/preprocess/test.csv', test_transformation)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=64, shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
    
    return train_dataloader, valid_dataloader, test_dataloader
