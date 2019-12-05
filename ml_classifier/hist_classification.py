# importing required libraries of opencv 
import cv2 
import glob
import numpy as np
# importing library for plotting 
from matplotlib import pyplot as plt 

#Get the files and iterate through each files.
path1 = 'grey_hair_mask'

file_list  = glob.glob(path1+'/*')

list1= []

#print(file_list)
for f in file_list:
    # reads an input image 
    img = cv2.imread(f,0) 

    # find frequency of pixels in range 0-255 
    histr = cv2.calcHist([img],[0],None,[256],[0,256])

    #print(histr[0:150].shape)
    #total no of pixels in that rigion where intensity is very high or white color.
    n = histr[80:210]

    #print("total pixels for black color intensity are ",total_pixels)

    sum_of_frequencies = total_pixel_in_mask = np.sum(n)

    #add the value of the gray color to get the intensity 

    break

    

        




#print(np.min(list1))
# mean = np.mean(list1) 
# median = np.median(list1)
# sum1 = np.sum(list1)
# print(mean,median,sum1/10000)
#print(sum1/10000)
