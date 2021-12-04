# -*- coding: utf-8 -*-
"""
@author: neoglez
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt  # for plotting
from NeuralAnthropometerDataset import (
    NeuralAnthropometerSyntheticImagesDatasetTrainTest,
)
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
import os
from NeuralAnthropometerTransform import TwoDToTensor
from NeuralAnthropometer import NeuralAnthropometer
import pandas as pd
import numpy as np
import locale

locale.setlocale(locale.LC_NUMERIC, "C")

rootDir = os.path.join("..", "..", "dataset")
rootDir = os.path.abspath(rootDir)

model_path = os.path.join(rootDir, "..", "model")
model_name = "Neural_Anthropometer_Model_04-02-2021_15-34-54.pt"
model_path = os.path.join(model_path, model_name)

transform = TwoDToTensor()

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = False
# create a new model, initialize random parameters
na = NeuralAnthropometer(debug=False).to(device)
na.load_state_dict(torch.load(model_path))
na.eval()

# loss and optimizer
# Note that the loss here is a scalar because reduction="mean",
# therefore, after subtracting element-wise the dimensions
# predicted - actuals, every element of the resulting tensor is squared.
# Finally, the tensor is flatten, the elements are summed and the sum is
# divided by the number of elements in the tensor.
# That means for us that we are going to test based on the mean squared error
# (squared L2 norm).
criterion = nn.MSELoss()

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

# No need these dataloaders for testing (reporting results)
# train_dt = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
# val_dt = DataLoader(val_dataset, shuffle=True, batch_size=batch_size)
test_dt = DataLoader(test_dataset, batch_size=1)

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

# Vector of loss to report. The loss is here the MSE of all HBDs.
loss_vector = []


for i, data in enumerate(test_dt, 1):
    # move to GPU if available
    actual_hbds = data["annotations"]["human_dimensions"].to(device)
    inputs = data["image"].to(device)
    # print("Images follow")
    # print(inputs)
    if debug:
        print("actual follows")
        print(actual_hbds)


    # predict and move to GPU
    predicted_hbds = na(inputs)  # .to(device)
    if debug:
        print("predicted follows")
        print(predicted_hbds)

    loss = criterion(predicted_hbds, actual_hbds)
    if debug:
        print("loss follows")
        print(loss)
    
    this_loss = loss.item()
    loss_vector.append(this_loss)
    print("[{}] loss: {}".format(data["subject_string"], this_loss))

print("Done Evaluating")
txt = "Average loss across all subjects is {:.2f} cm.".format(
    np.array(loss_vector).mean() * 100)
print(txt)

df = pd.DataFrame(data=loss_vector)
df.to_csv(
    "evaluating_losses.csv",
    index=False,
)
plt.plot(np.array(loss_vector), "r")
plt.title("Loss (MSE) per subject.\n{}".format(txt))
plt.xlabel("Subjects")
plt.ylabel("Loss")
plt.show()
