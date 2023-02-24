#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 00:31:51 2022

@author: eidos
"""

import os
import time
import numpy as np
from typing import Union

class FIFOInstance:
    def __init__(self, name: str):
        print(1)
        self.path_root = "/home/dell/meta_driving/fifo_space"
        self.path_pipe_srv_clt = self.path_root + "/" + "srv_2_clt.pipe"
        self.path_pipe_clt_srv = self.path_root + "/" + "clt_2_srv.pipe"
        # self.path_root = "/home/eidos/Workspace/GitKraken_ws/meta_driving/playground"
        # self.path_pipe_srv_clt = self.path_root + "/" + "dog.pipe"
        # self.path_pipe_clt_srv = self.path_root + "/" + "cat.pipe"
        print(2)
        if not os.path.exists(self.path_pipe_srv_clt):
            os.mkfifo(self.path_pipe_srv_clt)
        if not os.path.exists(self.path_pipe_clt_srv):
            os.mkfifo(self.path_pipe_clt_srv)
        print(3)
        if name == "server":
            print(4)
            self.wf = os.open(self.path_pipe_srv_clt, os.O_SYNC | os.O_WRONLY)
            print(5)
            self.rf = os.open(self.path_pipe_clt_srv, os.O_RDONLY)
        elif name == "client":
            self.rf = os.open(self.path_pipe_srv_clt, os.O_RDONLY)
            print(7)
            self.wf = os.open(self.path_pipe_clt_srv, os.O_SYNC | os.O_WRONLY)
            print(6)
            
            
        else:
            raise Exception
        
    def write(self, data: Union[str, bytes]):
        if isinstance(data, str):
            os.write(self.wf, data.encode())
        elif isinstance(data, bytes):
            os.write(self.wf, data)
    
    def read(self, n: int = 9999999):
        result = os.read(self.rf, n)
        return result
    
    def destroy(self):
        os.close(self.rf)
        os.close(self.wf)















