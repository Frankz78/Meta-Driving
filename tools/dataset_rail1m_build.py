#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 14:20:36 2022

@author: eidos
"""

import os 
import time

import cv2
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Only use wide_0_00xxx.jpg like images. Can be further increased in the future.
class PklDataset(Dataset):
    def __init__(self, main_path):
        # Root directory of all images
        self.main_path = main_path
        # Name of the json file
        self.name_json = 'data.json'
        # For save the reusable pkl file
        self.image_all_name = []
        try:
            # Contains the path to each image
            self.pkl_path = pd.read_pickle('/home/eidos/Workspace/CARLA/world_on_rails/leaderboard_model/dataset_path.pkl')
            self.pkl_path = self.pkl_path.iloc[0:50000]
        except:
            print('The pkl_path may not be generated.')
        
    def __getitem__(self, idx):
        # Get one image according to idx
        image_array = cv2.imread(self.pkl_path.iloc[idx, 0])/127.5 -1 
        
        return (image_array, image_array)
        
    def __len__(self):
        return len(self.pkl_path)
    
    def save_path_pkl(self):
        for _, dirs, _ in os.walk(self.main_path):
            break
        # Sort all names
        dirs.sort()
        count = 0
        for dir_name in dirs:
            count = count + 1
            time_start = time.time()
            print('Processing: ', dir_name)
            print('Progress: %i/%i'%(count, len(dirs)))
            # Open json file
            self.content = open(os.path.join(self.main_path, dir_name, self.name_json), 'r').read()
            # Read file in json format
            self.fold_json = json.loads(self.content)
            # The number of images in this folder
            image_num = self.fold_json['len'] - 5
            for i in range(image_num):
                # Get the relative path of one image
                image = self.fold_json['%i'%(i)]['wide_sem_0'].replace('_sem','').replace('png','jpg')
                self.image_all_name.append(os.path.join(self.main_path, dir_name, image.replace('/','',1)))
            
            time_end = time.time()
            print('time cost in this round= ', time_end - time_start)
        # Save this pkl file
        image_all_pd = pd.DataFrame(self.image_all_name)
        image_all_pd.to_pickle('dataset_path.pkl')
    
    
if __name__ =='__main__':
    originData_path = '/home/eidos/Workspace/CARLA/rails1M/main_trajs6_converted2'
    
    # dirs.sort()
    # a = dirs[0:1000]
    time_start = time.time()
    
    dataset = PklDataset(originData_path)
    dataset.save_path_pkl()
    time_end = time.time()
    print('time cost = ', time_end - time_start)
    
    
    
            
            
            
