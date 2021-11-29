# -*- coding: utf-8 -*-
"""
@author: neogl
"""
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt  # for plotting
from NeuralAnthropometerDataset import (
    NeuralAnthropometerSyntheticImagesDatasetTrainTest,
)
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
import os
from NeuralAnthropometerTransform import TwoDToTensor
from torchvision import transforms
from NeuralAnthropometer import NeuralAnthropometer
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime

transform = TwoDToTensor()

rootDir = os.path.join("..", "..", "dataset")
rootDir = os.path.abspath(rootDir)

# This will be very slow!!
batch_size = 1
# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = False
# create a new model, initialize random parameters
na = NeuralAnthropometer(debug=False).to(device)

# loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(na.parameters(), lr=0.005, momentum=0.9)

# 2700 instances to train and validate
na_train = NeuralAnthropometerSyntheticImagesDatasetTrainTest(
    rootDir, train=True, transform=transform
)
# 300 instances to test
na_test = NeuralAnthropometerSyntheticImagesDatasetTrainTest(
    rootDir, train=False, transform=transform
)
# train/val split:
train_size = int(0.8 * len(na_train))
val_size = len(na_train) - train_size

na_train, na_val = random_split(na_train, [train_size, val_size])
# assign to use in dataloaders...
train_dataset = na_train
val_dataset = na_val
test_dataset = na_test

print("Train dataset lenght is {}".format(len(train_dataset)))
print("Validation dataset lenght is {}".format(len(val_dataset)))
print("Test dataset lenght is {}".format(len(test_dataset)))

train_dt = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
val_dt = DataLoader(val_dataset, shuffle=True, batch_size=batch_size)
test_dt = DataLoader(test_dataset, batch_size=batch_size)

# training
# for batch_index, data in enumerate(train_dt):
# the structure of dict is:
# {'chest_circumference': tensor([1.0164], dtype=torch.float64),
#  'height': tensor([1.8133], dtype=torch.float64),
#  'inseam': tensor([0.8059], dtype=torch.float64),
#  'left_arm_length': tensor([0.5784], dtype=torch.float64),
#  'pelvis_circumference': tensor([1.0575], dtype=torch.float64),
#  'right_arm_length': tensor([0.6005], dtype=torch.float64),
#  'shoulder_width': tensor([0.4087], dtype=torch.float64),
#  'waist_circumference': tensor([0.8459], dtype=torch.float64)
#  }
# The equivalent tensor contains the information in the corresponding
# integer indices. It is important to remeber that all HBD are given in
# meters, so if you want to convert them to cm, you have to multiply by 100.
debug = False
average_epoch_loses = []
epochs = 10
for epoch in range(epochs):  # no. of epochs
    running_loss = 0
    for data in train_dt:
        # move to GPU if available
        actual_hbds = data["annotations"]["human_dimensions"].to(device)
        inputs = data["image"].to(device)
        # print("Images follow")
        # print(inputs)
        if debug:
            print("actual follows")
            print(actual_hbds)

        # set the parameter gradients to zero
        optimizer.zero_grad()

        # predict and move to GPU
        predicted_hbds = na(inputs)  # .to(device)
        if debug:
            print("predicted follows")
            print(predicted_hbds)

        loss = criterion(predicted_hbds, actual_hbds)
        if debug:
            print("loss follows")
            print(loss)
        # propagate the loss backward
        loss.backward()
        # update the gradients
        optimizer.step()
        this_loss = loss.item()
        print("Loss in this iteration follows")
        print(this_loss)
        running_loss += this_loss
        if debug:
            print("Running loss follows")
            print(running_loss)

    average_epoch_loss = running_loss / len(train_dt)
    average_epoch_loses.append(average_epoch_loss)
    print("[Epoch %d] loss: %.19f" % (epoch + 1, average_epoch_loss))

print("Done Training")
average_epoch_loses_filepath = os.path.join(
    rootDir, "log", "average_epoch_loses_{}.csv".format(1)
)
df = pd.DataFrame(data=average_epoch_loses)
df.to_csv(
    "average_epoch_loses.csv",
    index=False,
)
plt.plot(np.array(average_epoch_loses), "r")
