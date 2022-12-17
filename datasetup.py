import os
import numpy as np
import pandas as pd
import torch
import conf as cfg
from torch.utils.data import DataLoader, Dataset
from torchvision import io
from matplotlib import pyplot as plt
import cv2
from torch.utils.data import random_split
from torchvision import models



def transform(auto: bool=True):
    if auto:
        weights = models.ResNet50_Weights.DEFAULT
        t = weights.transforms()
        return t 



class CelebFace(Dataset):
    """
    doc
    """
    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path
        self.meta = self.img_name_cls()
        self.t = transform()
        
    def img_name_cls(self):
        meta = pd.read_csv(self.path['face_id'], sep=" ", header=None).values
        return meta

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        fname = self.meta[index][0]
        y = self.meta[index][1]
        x = cv2.imread(os.path.join(self.path['face'], fname))
        # x = torch.from_numpy(x).type(torch.float32).permute(2, 0, 1)
        x = self.t(torch.from_numpy(x).type(torch.float32).permute(2, 0, 1))
        return x, torch.tensor(y, dtype=torch.float32)

def split_and_create_train_val_test(dataset: Dataset, train_percent, batch_size):
    l = len(dataset)
    train_size = int(train_percent*l)
    evaluation_size = l - train_size
    validation_size = int(evaluation_size*0.8)
    test_size = evaluation_size - validation_size
    train, evaluation = random_split(dataset=dataset, lengths=[train_size, evaluation_size])
    validation, test = random_split(dataset=evaluation, lengths=[validation_size, test_size])
    train_loader = DataLoader(dataset=train, shuffle=True, batch_size=batch_size)
    validation_loader = DataLoader(dataset=validation, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(dataset=test, batch_size=100000)

    return train_loader, validation_loader, test_loader



def main():
    meta = pd.read_csv(cfg.paths['face_id'], sep=" ", header=None)

    dataset = CelebFace(path=cfg.paths)
    train, val, test = split_and_create_train_val_test(dataset=dataset, train_percent=0.1, batch_size=32)
    batch = next(iter(train))
    print(batch[0].shape)
    


if __name__ == "__main__":
    main()