#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 18:03:34 2022

@author: eidos
"""

import socket


class UDPServer:
    def __init__(self, server_port=12000):
        self.BUFSIZE = 2048
        self.server_port = server_port
        self.serverSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.serverSocket.bind(('', self.server_port))
        self.client_address = None
        print('This server is ready to receive')
        
    def receive(self) -> str:
        recv_raw_data, self.client_address = self.serverSocket.recvfrom(self.BUFSIZE)
        #print("Receive message: ", recv_raw_data.decode('utf-8'))
        return recv_raw_data.decode('utf-8')
    
    def send(self, send_data: str):
        if not type(send_data) == str:
            send_data = str(send_data)
            
        self.serverSocket.sendto(send_data.encode('utf-8'), self.client_address)

    def destroy(self):
        self.serverSocket.close()

if __name__ == "__main__":
    udp_server = UDPServer()
    recv_data = udp_server.receive()
    udp_server.send('123')
    udp_server.destroy()