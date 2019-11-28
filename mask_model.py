#Author Gaurav Kothyari 

import cv2
import numpy as np
import torch
import time
import os
import sys
import argparse
from PIL import Image
#from data import get_loader
import torchvision.transforms as std_trnsf
from utils import joint_transforms as jnt_trnsf

class Masking():
    def __init__(self,ckpt_dir,network):
        
        #self.img_f = img_file
        self.ckpt = ckpt_dir
        self.network = network
        self.device = 'cpu'
    
    def load_network(self):
        print("model loading --------")
        ckpt_dir = self.ckpt
        network = self.network
        device = self.device

        #check for the path 
        assert os.path.exists(ckpt_dir)


        # prepare network with trained parameters
        net = network.to(device)
        state = torch.load(ckpt_dir,map_location=torch.device('cpu'))
        net.load_state_dict(state['weight'])
        return net

    def get_mask(self, model, img_f):
        # get the prediction for the
        #model = self.load_network()

        ##transform the test image
        test_image_transforms = std_trnsf.Compose([
        std_trnsf.ToTensor(),
        std_trnsf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        img = Image.open(img_f)
        img = img.resize((600,600))
        img_cv = cv2.imread(img_f)
        img_cv = cv2.resize(img_cv,(600,600))
        data = test_image_transforms(img)
        data = torch.unsqueeze(data, dim=0)
        model.eval()
        data = data.to(self.device)

        # inference
        start = time.time()
        logit = model(data)
        duration = time.time() - start

        # prepare mask
        pred = torch.sigmoid(logit.cpu())[0][0].data.numpy()
        mh, mw = data.size(2), data.size(3)
        mask1 = pred >= 0.5
        # print(mask1)
        # print(type(pred))
        # print(pred.shape)

        b,g,r = cv2.split(img_cv)
        b[mask1 == True]= 255
        g[mask1 == True]= 255
        r[mask1 == True]= 255

        b[mask1 == False]= 0
        g[mask1 == False]= 0
        r[mask1 == False]= 0
        #mask3 = cv2.UMat(mask1)
        img3 = cv2.bitwise_and(img_cv,img_cv,mask = b)
        return img3
        
