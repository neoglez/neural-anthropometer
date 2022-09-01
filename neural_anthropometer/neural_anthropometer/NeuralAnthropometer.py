# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 16:58:12 2020

@author: neogl
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralAnthropometer(nn.Module):
    '''
    A Convolutional Neural Network (CNN) that is able to infer human body
        dimensions (HBD) from grayscale synthetic images.
    '''
    def __init__(self, debug=False):
        super().__init__()

        self.debug = debug
        self.conv1 = nn.Conv2d(1, 8, 5)
        self.bn = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 47 * 47, 84)
        self.fc2 = nn.Linear(84, 8)

    def forward(self, x):
        if self.debug:
            print("Shape of the input: ")
            print(x.shape)
            print("\n")

        x = self.bn(F.relu(self.conv1(x)))
        if self.debug:
            print("Shape of the data after the first convolution: ")
            print(x.shape)
            print("\n")

        x = self.pool1(x)
        if self.debug:
            print("Shape of the data after first pooling: ")
            print(x.shape)
            print("\n")

        x = F.relu(self.conv2(x))
        if self.debug:
            print("Shape of the data after the second convolution: ")
            print(x.shape)
            print("\n")

        x = self.pool2(x)
        if self.debug:
            print("Shape of the data after second pooling: ")
            print(x.shape)
            print("\n")

        x = x.view(-1, self.num_flat_features(x))
        if self.debug:
            print("Shape of the data before entering the fully connected: ")
            print(x.shape)
            print("\n")

        x = F.relu(self.fc1(x))
        if self.debug:
            print("Shape of the data after the hidden layer: ")
            print(x.shape)
            print("\n")

        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features