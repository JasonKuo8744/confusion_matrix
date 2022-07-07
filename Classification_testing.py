 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 12:13:25 2021

@author: coco
"""

#%% Import
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from os.path import join
from random import sample
from PIL import Image


from keras.layers import Input, Conv2D,Conv2DTranspose
from keras.layers import UpSampling2D, Dense, Flatten
from keras.layers import MaxPooling2D, Reshape
from keras.layers import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.models import load_model

#%% Define

def test(pic_path):
    CAEmodel = load_model(join(os.getcwd(),'model_LS.h5'))
    #CAEmodel = load_model(join(os.getcwd(),'CAEmodel_LS_1.h5'))
    CAEmodel.compile(loss='categorical_crossentropy', optimizer=Adam(0.0002, 0.5))
    
    files = os.listdir(pic_path)
    index = sample(range(0, len(files)), 20)
    name = files[index[0]]
    picture = np.load(join(pic_path,name))
    test_data = np.zeros((1,896,896,1))
    test_data[0,:,:,0] = picture
    Range_predict = CAEmodel.predict(test_data)
    
    return name,Range_predict


def test2(pic_path,name):
    #CAEmodel = load_model(join(os.getcwd(),'model_LS.h5'))
    CAEmodel = load_model(join(os.getcwd(),'model_LS.h5'))
    CAEmodel.compile(loss='categorical_crossentropy', optimizer=Adam(0.0002, 0.5))

    picture = np.load(join(pic_path,name))
    test_data = np.zeros((1,896,896,1))
    test_data[0,:,:,0] = picture
    Range_predict = CAEmodel.predict(test_data)
    
    return Range_predict

def mid_layer_model(model,layer_name):
    input_layer = model.input
    output_layer = model.get_layer(layer_name).output
        
    return Model(inputs=input_layer, outputs=output_layer)

#%%

#picture = np.load("E:\\LS\\NPY\\Random_Pop4766.npy")
#plt.imshow(picture)

start = time.time()

test_pic_path = r"E:\LS_test_data\test_1"
test_label_path = r"E:\LS_test_data\test_1_label"



#test_pic_path = join(os.getcwd(),'testing_LS_NPY')    
#test_label_path = join(os.getcwd(),'test_label')    

files = os.listdir(test_pic_path)
#index = sample(range(0, len(files)), 20)  
   
#SRAF_name = 'First_Pop'
#SRAF_name = 'Random_pop'
index = np.arange(5)   
number = len(index)
result = np.zeros([number,3])

for i in range(number):
    #name = SRAF_name + str(index[i]) + '.npy'
    name = files[index[i]]
    Range_predict = test2(test_pic_path,name)
    p_row , p_col = np.where(Range_predict == np.max(Range_predict))
    
    label_name = 'label_' + name
    label = np.load(join(test_label_path,label_name))
    label_t = np.transpose(label)
    l_row , l_col = np.where(label_t == np.max(label_t))
    
    result[i,0] = l_col
    result[i,1] = p_col
    
    if p_row == l_row and p_col == l_col:
        result[i,2] = 1
    else:
        result[i,2] = 0


end = time.time()
cost_time = end-start
print(cost_time)

answer =  np.sum(result[:,2])
print(answer)


# Confusion matrix

TP_r = 0
FP_r = 0
FN_r = 0
TN_r = 0
total = 0
interval = 4

for i in range(len(result)):
    
    if result[i,0] == interval:
        total = total+1
    
    # Correct prediction
    if result[i,0] == interval and result[i,2] == 1:
        TP_r = TP_r+1
    
    # This interval, wrong prediction
    if result[i,0] == interval and result[i,2] == 0:
        FP_r = FP_r+1
    
    # Not this interval, correct prediction
    if result[i,0] != interval and result[i,2] == 1:
        TN_r = TN_r+1
        
    # Not this interval, predict to this interval
    if result[i,0] != interval and result[i,1] == interval:
        FN_r = FN_r+1



    
    
    
  
    
    
        
    
    




