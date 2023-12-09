
from lavis.models import model_zoo

import re
import torch
from torch import nn
from torchvision import transforms

import json

from PIL import Image

import cv2
import numpy as np

from skimage import transform as skimage_transform
from scipy.ndimage import filters
from matplotlib import pyplot as plt

import torch
from PIL import Image
import requests
from lavis.models import load_model_and_preprocess, load_model
import argparse


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Mask():
    
    
    def __init__(self):
        
        
        self.model, self.vision_processer, _ = load_model_and_preprocess(name="blip2", model_type="pretrain")
        self.model.to(device)

        # save attention gradients
        self.model.Qformer.base_model.base_model.encoder.layer[8].crossattention.self.save_attention = True
        self.model.Qformer.base_model.base_model.encoder.layer[8].crossattention.self.save_attention
    
        
    

    def pre_caption(self, caption,max_words=30):
        caption = re.sub(
            r"([,.'!?\"()*#:;~])",
            '',
            caption.lower(),
        ).replace('-', ' ').replace('/', ' ')

        caption = re.sub(
            r"\s{2,}",
            ' ',
            caption,
        )
        caption = caption.rstrip('\n') 
        caption = caption.strip(' ')

        #truncate caption
        caption_words = caption.split(' ')
        if len(caption_words)>max_words:
            caption = ' '.join(caption_words[:max_words])            
        return caption


    def getAttMap(self, img, attMap, blur = True, overlap = True):
        attMap -= attMap.min()
        if attMap.max() > 0:
            attMap /= attMap.max()
        attMap = skimage_transform.resize(attMap, (img.shape[:2]), order = 3, mode = 'constant')
        if blur:
            attMap = filters.gaussian_filter(attMap, 0.02*max(img.shape[:2]))
            attMap -= attMap.min()
            attMap /= attMap.max()
        cmap = plt.get_cmap('jet')
        attMapV = cmap(attMap)
        attMapV = np.delete(attMapV, 3, 2)
        if overlap:
            attMap = 1*(1-attMap**0.7).reshape(attMap.shape + (1,))*img + (attMap**0.7).reshape(attMap.shape+(1,)) * attMapV
        return attMap


    def visualize(self, text_tokens, image_path, gradcam):
        num_image = len(text_tokens.input_ids[0]) 
        fig, ax = plt.subplots(num_image, 1, figsize=(15,5*num_image))

        rgb_image = cv2.imread(image_path)[:, :, ::-1]
        rgb_image = np.float32(rgb_image) / 255

        np_gradcam = gradcam.cpu().numpy()

        ax[0].imshow(rgb_image)
        ax[0].set_yticks([])
        ax[0].set_xticks([])
        ax[0].set_xlabel("Image")

        for i,token_id in enumerate(text_tokens.input_ids[0][1:]):
            word = self.model.tokenizer.decode([token_id])
            gradcam_image = getAttMap(rgb_image, np_gradcam[i+1], blur=False)
            ax[i+1].imshow(gradcam_image)
            ax[i+1].set_yticks([])
            ax[i+1].set_xticks([])
            ax[i+1].set_xlabel(word)


            
    def compute_mask(self, path_to_image, prompt, output_path):

        # process image
        raw_image = Image.open(path_to_image)
        image = self.vision_processer["eval"](raw_image).unsqueeze(0).to(device)

        # assemble inputs
        pre_text = self.pre_caption(prompt)
        samples = {'image': image.to(device), 'text_input': pre_text}

        # only itc
        output = self.model(samples)
        loss = output['loss']
    
        # tokenize text
        text_tokens = self.model.tokenizer(
            pre_text,
            padding="max_length",
            truncation=True,
            max_length=32,
            return_tensors="pt",
        ).to(device)
    
    
        self.model.zero_grad()
        loss.backward()
    
        # compute attention map
        with torch.no_grad():

            # for each token
            token_mask = text_tokens.attention_mask.view(text_tokens.attention_mask.size(0),1,-1,1,1)

            print("mask shape: ", token_mask.size())

            # text_encoder here is a pre-trained bert
            gradients=self.model.Qformer.base_model.base_model.encoder.layer[8].crossattention.self.get_attn_gradients()
            # att_map
            att_map=self.model.Qformer.base_model.base_model.encoder.layer[8].crossattention.self.get_attention_map()

            att_map = att_map[:, :, :, 1:].reshape(image.size(0), 12, -1, 16, 16) * token_mask # 12 for the 12 layers
            gradients = gradients[:, :, :, 1:].clamp(0).reshape(image.size(0), 12, -1, 16, 16) * token_mask

            # back pass through gradients
            att_map = att_map * gradients
            att_map = att_map[0].mean(0).cpu().detach()

            mask = np.mean(att_map.cpu().detach().numpy(), axis=0)

            # produce binary mask
            mask[mask < np.mean(mask)] = 0
            mask[mask > np.mean(mask)] = 1

            imgplot = plt.imshow(mask, cmap='gray')
            plt.axis('off')
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)

    


        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_image", type=str, help='target image')
    parser.add_argument("prompt", type=str, help='text prompt')
    parser.add_argument("output_path", type=str, help='output_path of the mask')
    args = parser.parse_args()
    
    
    mask_model = Mask()
    mask_model.compute_mask(args.path_to_image, args.prompt, args.output_path)
    
    
