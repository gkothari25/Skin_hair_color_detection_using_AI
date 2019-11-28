#!/usr/bin/env python
#author Gaurav Kothyari
#The MIT License (MIT)
#Copyright (c) 2019 @Kothari Brothers

#In this example I use the range detector class to detect skin in two pictures
import sys
import os
import dlib
import glob
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from color_detection import RangeColorDetector 
import mask_model
from networks import pspnet
import prediction_mask

import warnings
warnings.simplefilter("ignore", UserWarning)

#get the image and pass this through our model and get the bbox of our image.
predictor_path = 'shape_predictor_68_face_landmarks.dat'
faces_folder_path = 'test'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

#get the mask of the hair from the full image.
network = pspnet.PSPNet(num_class=1, base_network='resnet101')
ckptr = 'ckpt/pspnet_resnet101_sgd_lr_0.002_epoch_100_test_iou_0.918.pth'

#call the network class and get the hair mask
mm = mask_model.Masking(ckptr,network)
model = mm.load_network()


for f in glob.glob(os.path.join(faces_folder_path, "*.*")):
    print("Processing file: {}".format(f))
    img = cv2.imread(f)

    #Ask the detector to find the bounding boxes of each face. The 1 in the
    #second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,500)
    fontScale              = 8
    fontColor              = (255,255,255)
    lineType               = 2

    #this iteration for multiple images
    try:
        d = dets[0]
    
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        
        # Draw the face landmarks on the screen.
        cv2.rectangle(img,(d.left(),d.top()),(d.right(),d.bottom()),color=(255,0,0),thickness=6, lineType=8, shift=0)
        #win.add_overlay(shape)

        #crop the face 
        cropped_rgb = img_1 = img[d.top():d.bottom(),d.left(): d.right()]
        #cropped_hsv = cv2.cvtColor(cropped_rgb,cv2.COLOR_BGR2HSV)
        cropped_rgb1 = cv2.resize(cropped_rgb,(800,800))

        #noise removal is a key in hsv domain 
        # saturation will decide the black color skin /how greish or dark this color is
        #First image boundaries
        min_range = np.array([1, 35, 130], dtype = "uint8") #lower HSV boundary of skin color
        max_range = np.array([50, 100, 255], dtype = "uint8") #upper HSV boundary of skin color
        
        my_skin_detector = RangeColorDetector(min_range, max_range) #Define the detector object
        
        image = img   #cv2.imread("too_w.jpg") #Read the image with OpenCV  

        #We do not need to remove noise from this image so morph_opening and blur are se to False
        image_filtered ,mask = my_skin_detector.returnFiltered(cropped_rgb1, morph_opening=False, blur=False, kernel_size=3, iterations=1)
        contours, hierarchy = cv2.findContours(image = mask,  mode = cv2.RETR_EXTERNAL,method = cv2.CHAIN_APPROX_SIMPLE)
        
        area = []        
        
        for i in contours:
            area.append(cv2.contourArea(i))
        
        max_skin_area = max(area)

        img_label = cv2.imread('Background.jpg')
        #drawing critera
        # font 
        font = cv2.FONT_HERSHEY_PLAIN 

        # org 
        org = (20,500) 
        
        # fontScale 
        fontScale = 4

        # Blue color in BGR 
        color = (255,0, ) 
        
        # Line thickness of 2 px 
        thickness = 4
        
        #decision point by area
        if max_skin_area > 4000 and max_skin_area < 50000.0:
            text = 'Skin is Dark'
            cv2.putText(img_label, text, org, font,fontScale, color, thickness, cv2.LINE_AA) 
        
        elif max_skin_area < 4000.0:
            text = 'Skin is Brown'
            cv2.putText(img_label, text, org, font,fontScale, color, thickness, cv2.LINE_AA)
        
        elif max_skin_area > 50000.0:
            text = 'Skin is White'
            cv2.putText(img_label, text, org, font,fontScale, color, thickness, cv2.LINE_AA) 
    

        hair_mask = mm.get_mask(model, f)
        
        #pass the cropped image to prediction class and get the prediction.
        pred_color = prediction_mask.Hairprediction(hair_mask)

        result_text = pred_color.get_color_prediction()

        #dimesions of text
        text = result_text

        org = (20,int(img_label.shape[1]/2)) 
        
        cv2.putText(img_label, text, org, font,fontScale, color, thickness, cv2.LINE_AA) 
        
        plt.subplot(1, 3, 1), plt.imshow(img[:,:,::-1])
        plt.subplot(1, 3, 2), plt.imshow(cropped_rgb1[:,:,::-1])
        plt.subplot(1, 3, 3), plt.imshow(img_label)
        
        temp = f.split('/')[-1].split('.')[0]
        path = 'output/{}'.format(temp) 
        plt.savefig(path)
    
    except Exception as E:
        print('No Face detected Please feed the clear image with face ')


