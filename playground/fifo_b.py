#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 00:43:45 2022

@author: eidos
"""
import numpy as np
from fifo_instance import FIFOInstance

b = FIFOInstance('client')

array = np.array(range(512)).reshape(2,256)
array = array.tobytes()

b.write(array)
feedback = b.read()
print(feedback)
b.destroy()