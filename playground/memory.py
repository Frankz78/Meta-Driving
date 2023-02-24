#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 23:24:34 2022

@author: eidos
"""
import _ctypes
import numpy as np

a = np.array(range(512)).reshape(2,256)
print(id(a))
print(a.ctypes.data)

b = _ctypes.PyObj_FromPtr(id(a))

import ctypes

# c = ctypes.cast(int(a), ctypes.py_object).value