#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 23:51:22 2022

@author: eidos
"""

import os
import time
import uuid
import cv2
import numpy as np

read_path = "./cat.pipe"
write_path = "./dog.pipe"

# if os.path.exists(write_path):
#     os.remove(write_path)
#     os.mkfifo(write_path)
#     os.remove(read_path)
#     os.mkfifo(read_path)

wf = os.open(write_path, os.O_SYNC | os.O_CREAT | os.O_RDWR)
rf = os.open(read_path, os.O_RDONLY)


    
s = "test_2"
print('before write')
os.write(wf, s.encode())
print('before read')
result = os.read(rf, 100)
print(result)

os.close(rf)
os.close(wf)