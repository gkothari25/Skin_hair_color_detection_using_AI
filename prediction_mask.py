#Author Gaurav Kothyari 

import numpy as np
import PIL
import cv2
from PIL import Image
import pickle


class Hairprediction():

    def __init__(self,mask_img):
        self.mask_img = mask_img


    def get_color_prediction(self):

        mask = self.mask_img 

        label = {0:'Black',1:'Grey',2:'Blonde'}

        # find frequency of pixels in range 0-255 
        histr = cv2.calcHist([mask],[0],None,[256],[0,256]) 

        histr1 = histr[200:255]

        col = [x for x in range(5)]

        meanp = np.mean(histr1)
        medianp = np.median(histr1)
        n_pixels = np.sum(histr1)
        stdp =    np.std(histr1)
        varp =    np.var(histr1)

        #print(varp)
        row1 = np.array([meanp,medianp,n_pixels,stdp,varp])
        row2 = np.reshape(row1,(1,-1))

        #scaler load
        scaler_path = 'scaler_final_new.pkl'
        scaler = pickle.load(open(scaler_path,'rb'))
        col_norm = scaler.transform(row2)

        #model load
        model_path = 'split_model_new.pkl'
        model = pickle.load(open(model_path,'rb'))
        pred = model.predict(col_norm)

        text = 'Hair color is {}'.format(label[int(pred)])
        return text
        
