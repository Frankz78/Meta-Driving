#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 19:57:57 2022

@author: eidos
"""

import socket


class UDPClient:
    def __init__(self, server_ip: str = '127.0.0.1', server_port: int = 12000):
        self.BUFSIZE = 2048
        self.server_ip = server_ip
        self.server_port = server_port
        self.clientSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    def receive(self) -> str:
        # server_address is useless here
        recv_raw_data, server_address = self.clientSocket.recvfrom(self.BUFSIZE)
        # print("Receive message is: ", recv_raw_data.decode('utf-8'))
        return recv_raw_data.decode('utf-8')
    
    def send(self, send_data: str):
        if not type(send_data) == str:
            send_data = str(send_data)
            
        self.clientSocket.sendto(send_data.encode('utf-8'), (self.server_ip, self.server_port))
        
    def destroy(self):
        self.clientSocket.close()
        
if __name__ == "__main__":
    udp_client = UDPClient()
    udp_client.send('456')
    recv_data = udp_client.receive()
    udp_client.destroy()