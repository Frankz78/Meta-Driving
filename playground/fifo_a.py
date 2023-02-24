#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 00:42:57 2022

@author: eidos
"""
import numpy as np
from fifo_instance import FIFOInstance

a = FIFOInstance('server')
print("finish initialize")
feedback = a.read()
result = np.frombuffer(feedback, dtype=np.int64).reshape(2, 256)
print(result)


a.write('a_msg')
a.destroy()


