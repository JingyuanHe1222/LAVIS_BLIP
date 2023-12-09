
from lavis.models import model_zoo

import re
import torch
from torch import nn
from torchvision import transforms

import json
import pickle
from PIL import Image

import cv2
import numpy as np

from skimage import transform as skimage_transform
from scipy.ndimage import filters
from matplotlib import pyplot as plt

import requests
from lavis.models import load_model_and_preprocess, load_model

from tqdm import tqdm

from CustomDataset import CustomDataset
from torch.utils.data import DataLoader

# model optim
from torch.optim import AdamW, SGD

# lr schedulers
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the blip2 qformer model only lang_processer has no use, model has its own tokenizer
model, vision_processer, lang_processer = load_model_and_preprocess(name="blip2", model_type="pretrain")
model.to(device)

dataset = CustomDataset("/home/ubuntu/Text_Guided_Image_Editing/data") # hard-coded path
print(dataset[0])

train_dataloader = DataLoader(dataset, shuffle=True, batch_size=1)

epoch=25
lr=1e-5
eps=5e-5

model.optimizer = AdamW(model.parameters(), lr=lr, eps=eps)

train_steps = epoch*len(train_dataloader)
warm_step = int(train_steps*0.1)
model.scheduler = get_cosine_schedule_with_warmup(model.optimizer, num_warmup_steps=warm_step,num_training_steps=train_steps)

# Fine Tuning
all_loss = []

for e in tqdm(range(epoch)):
    
    losses = 0
    for image, label, text in dataset:
        
        image = vision_processer["eval"](image).unsqueeze(0)
        
        samples = {'image': image.to(device), 'text_input': text}
        
        output = model(samples)
        loss = output['loss']
        
        losses += loss.item()
        
        model.zero_grad()
        loss.backward()
        
        model.optimizer.step()  # backprop to update the weights

        if model.scheduler is not None:
            model.scheduler.step()  # update learning rate schedule 
    
    print("epoch: ", e, "losses: ", losses/len(dataset))
    all_loss.append(losses/len(dataset))
        

# dump the loss stats
with open("tune_losses", 'wb') as f:
    pickle.dump(all_loss, f)

# make a plot for report
plt.plot(all_loss)
plt.xlabel("Epoch")
plt.ylabel("Loss")






