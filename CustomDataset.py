import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, utils
import torchvision.datasets as datasets

class CustomDataset(Dataset):

    def __init__(self, path):

        self.root_dir = path

        if not os.path.exists(path):
            print("dataset path does not exist")
            return 

        # dataset
        self.image_dataset = []
        self.classes = []
        self.caption = []

        for level in ["Easy", "Mid", "Hard"]:
            data_file = os.path.join(self.root_dir, level, "data.txt")
            with open(data_file, 'r') as f:
                for line in f:
                    info = line.split(',')
                    img_path = os.path.join(self.root_dir, level, info[0])
                    image = Image.open(img_path).convert("RGB")
                    self.image_dataset.append(image)
                    self.classes.append(info[1])
                    caption = info[2]
                    if info[2][-1] == '\n':
                        caption = info[2][:-1]
                    self.caption.append(caption) # get rid of 
        print("dataset built with", len(self.classes), "instances")        

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        return self.image_dataset[idx], self.classes[idx], self.caption[idx]