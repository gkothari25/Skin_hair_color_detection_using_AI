import cv2
import pickle
import numpy as np

mask = '/home/gaurav/face/pytorch-hair-segmentation/test_report/69.jpg'

label = {0:'black',1:'grey',2:'blonde'}

# reads an input image 
img = cv2.imread(mask,0) 

# find frequency of pixels in range 0-255 
histr = cv2.calcHist([img],[0],None,[256],[0,256]) 

#print(histr[0:150].shape)

histr1 = histr[200:255]
#total_pixels_inthatrange = np.sum(histr[200:255])
# median_pixel = np.median(histr)
col = [x for x in range(5)]
# if total_pixels_inthatrange < 143 or 

meanp = np.mean(histr1)
medianp = np.median(histr1)
n_pixels = np.sum(histr1)
stdp =    np.std(histr1)
varp =    np.var(histr1)

#print(varp)
row1 = np.array([meanp,medianp,n_pixels,stdp,varp])
row2 = np.reshape(row1,(1,-1))

#scaler load
scaler_path = 'best_model/scaler_final_new.pkl'
scaler = pickle.load(open(scaler_path,'rb'))
col_norm = scaler.transform(row2)

#model load
model_path = 'best_model/split_model_new.pkl'
model = pickle.load(open(model_path,'rb'))
pred = model.predict(col_norm)

print(pred)
#print("color is ",label[int(pred)])

