#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 14:50:40 2022

@author: eidos
"""
# import os
# import sys
# import _ctypes
import numpy as np
import traceback
from gym import Env, spaces
import pyglet

import os
import sys
top_path = '/home/dell/meta_driving' 
if not top_path in sys.path:
    sys.path.append(top_path)
    
from tools.udp_server import UDPServer
from tools.fifo_instance import FIFOInstance

import torch
import torch.nn as nn
import torch.nn.functional as F


class Extraction(Env):
    def __init__(self):
        self.index = 0
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low = np.zeros(2048).reshape(2,-1), 
                                            high = np.zeros(2048).reshape(2,-1), 
                                            shape=(2, 1024))
        # mean and variance 
        # self.state = np.zeros(1024).reshape(2, 1024)
        # self.state = None
        
        # Initialize the socket
        self.udp_server_list = []
        for i in range(6):
            self.udp_server_list.append(UDPServer(server_port=12000+i))
        # Initialize the FIFO server
        self.udp_receive_RouteComple = UDPServer(server_port=12006)
        self.udp_receive_Timeout = UDPServer(server_port=12007)
        self.fifo_server = FIFOInstance('server')
        self.first_state = None
        self.counter = 0
        self.FailedTest = [0,0,0,0,0,0]
        self.ScoreTest = [0,0,0,0,0,0]
        self.Failed_counter = [0,0,0,0,0,0]
        self.counting_starter = ['end','end','end','end','end','end']
        self.RouteCompletion = 0
        self.Timeout = None

        # # Trigger the first read
        # for udp_server in self.udp_server_list:
        #     udp_server.receive()
        # # Trigger the first round
        # self.first_state = self.fifo_server.read()
        
        
    def step(self, action):
        # Initialization
        done = False
        reward = 0
        infractions = []
        score = 0
        threshold_send = 100
        episode_penalty = 0
            
        if self.counter % threshold_send == 0:
            action = 1  # specific action to perform
        self.counter += 1
        if self.counter >= threshold_send:
            self.counter = 0


        # To do
        # Based on the action, send the image to the CARLA AI
        # <====================================================================
        # Give action
        self.fifo_server.write('%d'%action)

        # Carla need to go to the next frame
        #print('Wait for client')
        # Get state
        raw_state = self.fifo_server.read()
        # If the CARLA server is sto
        try:
            n_state = np.frombuffer(raw_state, dtype = np.float32).reshape(2, 1024)
            self.previous_raw_state = raw_state
            #print('Wrong is Right!')
            #print(n_state)
        except ValueError:
            traceback.print_exc()
            n_state = np.frombuffer(self.previous_raw_state, dtype = np.float32).reshape(2, 1024)
            done = True

        # Get reward
        for udp_server in self.udp_server_list:
            infractions.append(udp_server.receive())
        #print(infractions)
        
        #lane = infractions[0].split('_')
        #print(float(lane[4]))     
        
        #reward = function(communication(action))
        #cost = function(infractions)
        
        if action == 1:
            reward = -1
            #score = -(0.5*int(infractions[0][2])+0.6*int(infractions[1][2])+0.7*int(infractions[2][2])+0.8*int(infractions[3][2])+0.9*int(infractions[4][2])+1.0*int(infractions[5][2]))
            #score[0] = -200        
            #score = torch.Tensor(score)
            #cost = torch.clamp(m(score),-1,0).tolist()
            
            #print(reward)
        
        elif action == 0:
            reward = 1
        else:
            print('wrong_action')
        
        for n in range(1,6):
            self.FailedTest[n] += int(infractions[n][2])
            if int(infractions[n][2]) == 1 and self.Failed_counter[n] == 0:
                self.counting_starter[n] = 'start'
            if self.counting_starter[n] == 'start':
                self.Failed_counter[n] += 1
                if self.Failed_counter[n] == 1:
                    self.ScoreTest[n] = 1
                elif self.Failed_counter[n] == 9:
                    self.Failed_counter[n]= 0
                    self.counting_starter[n] = 'end'
                    
        score = 100 * (
            0.6*self.ScoreTest[1]+
            0.7*self.ScoreTest[2]+
            0.7*self.ScoreTest[3]+
            0.8*self.ScoreTest[4]+
            0.7*self.ScoreTest[5])
        
        lane = infractions[0].split('_')
        devi = float(lane[4])
        
        if done:
            self.RouteCompletion = self.udp_receive_RouteComple.receive()
            self.Timeout = self.udp_receive_Timeout.receive()
            # print(float(self.RouteCompletion.split(' ')[0]))
            # print(self.Timeout)
            if float(self.RouteCompletion.split(' ')[0]) < 95 and self.Timeout == 'timeout':
                episode_penalty = 100
        penalty = devi * 0.5 + score + episode_penalty
        # print(penalty)
        # print(self.FailedTest)
        # print(self.ScoreTest)
        self.ScoreTest = [0,0,0,0,0,0]
        if done:
            self.FailedTest = [0,0,0,0,0,0]
        return np.array(n_state, dtype=np.float32), reward, penalty, done, {}
        
    def reset(self):
        # Trigger the first read
        #print('UDP_start')
        if self.first_state is None:
            for udp_server in self.udp_server_list:
                udp_server.receive()
        else:
            for udp_server in self.udp_server_list:
                udp_server.destroy()
            self.udp_server_list = []
            for i in range(6):
                self.udp_server_list.append(UDPServer(server_port=12000+i))
            
        # Trigger the first round
        #print('FIFO_start')
        self.first_state = self.fifo_server.read()
        #print('FIFO_over')
        state = np.frombuffer(self.first_state, dtype = np.float32).reshape(2, 1024)
        return np.array(state, dtype=np.float32)


    def render(self):
        pass
    
    def destroy(self):
        self.fifo_server.destroy()
        for udp_server in self.udp_server_list:
            udp_server.destroy()
        
        
class Viewer(pyglet.window.Window):
    def __init__(self):
        pass
        
        
    def render(self):
        pass
        
        
    def ondraw(self):
        pass
        
        
    def update_view(self):
        pass
        

