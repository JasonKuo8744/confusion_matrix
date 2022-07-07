#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 17:40:30 2022

@author: coco
"""

import os
import numpy as np
from os.path import join
import matplotlib.pyplot as plt

path = "E:\\LS_test_data\\test_1_result"
data = np.load(join(path,'prediction_record_new.npy'))

TP_r = 0
FP_r = 0
FN_r = 0
TN_r = 0
total = 0
interval = 4

for i in range(len(data)):
    
    if data[i,0] == interval:
        total = total+1
    
    # Correct prediction
    if data[i,0] == interval and data[i,2] == 1:
        TP_r = TP_r+1
    
    # This interval, wrong prediction
    if data[i,0] == interval and data[i,2] == 0:
        FP_r = FP_r+1
    
    # Not this interval, correct prediction
    if data[i,0] != interval and data[i,2] == 1:
        TN_r = TN_r+1
        
    # Not this interval, predict to this interval
    if data[i,0] != interval and data[i,1] == interval:
        FN_r = FN_r+1

