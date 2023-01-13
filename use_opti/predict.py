# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 15:29:48 2021

@author: guiguidxhb
"""

from my_srad import smooth
from unet import UNet
from DR_unet import DR_UNet
import logging
import torch
import numpy as np
from utils.dataset import BasicDataset
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

from debug_tool import *

def initial(Model):
    net = DR_UNet(n_channels=1, n_classes=1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(Model, map_location=device))
    logging.info("Loading model {}".format(Model))
    return net

def initial_xz(Model):
    net = DR_UNet(n_channels=1, n_classes=3)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(Model, map_location=device))
    logging.info("Loading model {}".format(Model))
    return net

def initial_flip(Model):
    net = UNet(n_channels=1, n_classes=1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(Model, map_location=device))
    logging.info("Loading model {}".format(Model))
    return net

def segment(full_img, net):
    net.eval()
    scale_factor = 0.5
    out_threshold = 0.5
    
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    
    output = net(img)
    probs = torch.sigmoid(output)
    probs = probs.squeeze(0)

    tf = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(full_img.size[1]),
            transforms.ToTensor()
        ]
    )

    probs = tf(probs.cpu())
    full_mask = probs.squeeze().cpu().numpy()
    mask = full_mask > out_threshold
    
    #mask_img = Image.fromarray((mask * 255).astype(np.uint8))    
    return mask

def segment_xz(full_img, net):
    net.eval()
    scale_factor = 0.5
    out_threshold = 0.5
    
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    
    output = net(img)
    probs = F.softmax(output, dim=1)
    probs = probs.squeeze(0)

    tf = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(full_img.size[1]),
            transforms.ToTensor()
        ]
    )

    probs = tf(probs.cpu())
    full_mask = probs.squeeze().cpu().numpy()
    full_mask = full_mask.transpose((1, 2, 0))
    mask = full_mask > out_threshold
    
    mask = mask[:,:,1]
    #mask_img = Image.fromarray((mask * 255).astype(np.uint8))    
    return mask
    
def flip_predict(full_img, net):
    net.eval()
    scale_factor = 0.5
    out_threshold = 0.5
    
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    
    output = net(img)
    probs = torch.sigmoid(output)
    probs = probs.squeeze(0)

    tf = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(full_img.size[1]),
            transforms.ToTensor()
        ]
    )

    probs = tf(probs.cpu())
    full_mask = probs.squeeze().cpu().numpy()
    mask = full_mask > out_threshold
    
    #mask_img = Image.fromarray((mask * 255).astype(np.uint8))    
    return mask


def cut_and_smooth(img, net, slc_reverse , root_path , filename):
    if os.path.exists(root_path + filename +'/mask_result/' + str(slc_reverse) + '.png'):
        mask = Image.open(root_path + filename +'/mask_result/' + str(slc_reverse) + '.png')
    else:
        mask = segment(img, net)
    # img_save = Image.fromarray(img_array.astype(np.uint8))
        mask = Image.fromarray((mask * 255).astype(np.uint8))    
    #mask.save(root_path + filename +'/mask_result/' + str(slc_reverse) + '.png')        #存储分割后的mask图
    
    # AR_img = np.array(img) * mask
    # AR_img = Image.fromarray(AR_img.astype(np.uint8)) 
    # mask = segment_xz(AR_img, net_xz)
    # mask = Image.fromarray((mask * 255).astype(np.uint8))   
    # mask.save(root_path + filename +'/no_xz_result/' + str(slc_reverse) + '.png')        #存储分割后的mask图
    
    '''裁剪'''
    box = (112,112,399,399)
    newW = 512
    newH = 512
    
    img = img.crop(box)
    img = img.resize((newW, newH))
    
    mask = mask.crop(box)
    mask = mask.resize((newW, newH))
    
    '''平滑'''
    smooth_img = smooth(img,10)
    
    '''保留分割部分'''
    mask_array = np.array(mask)
    img_array = np.array(smooth_img)
    mask_normalize = mask_array//255
    maintain_img = mask_normalize * img_array
    
    return maintain_img , mask_array

    