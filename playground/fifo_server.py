#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 23:51:07 2022

@author: eidos
"""

import os
import time
import numpy as np


# read_path = "/home/eidos/Workspace/GitKraken_ws/meta_driving/fifo_space/srv_2_clt.pipe"
# write_path = "/home/eidos/Workspace/GitKraken_ws/meta_driving/fifo_space/clt_2_srv.pipe"
read_path = "./dog.pipe"
write_path = "./cat.pipe"

# # if os.path.exists(write_path):
# os.remove(write_path)
# os.mkfifo(write_path)
# os.remove(read_path)
# os.mkfifo(read_path)

print("before open rf")
rf = os.open(read_path, os.O_RDONLY)
print("before open wf")
wf = os.open(write_path, os.O_SYNC | os.O_CREAT | os.O_RDWR)

    
s = "test_1"
print('before write')
os.write(wf, s.encode())
print('before read')
result = os.read(rf, 100)
print(result)

os.close(rf)
os.close(wf)
