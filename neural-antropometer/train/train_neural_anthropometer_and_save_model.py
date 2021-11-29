import torch
import torch.nn as nn
import torch.optim as optim
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
import datetime

transform = TwoDToTensor()

rootDir = os.path.join("..", "..", "dataset")
rootDir = os.path.abspath(rootDir)

model_path = os.path.join(rootDir, "..", "model")
batch_size = 100
# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = False
# create a new model, initialize random parameters
na = NeuralAnthropometer(debug=False).to(device)

# loss and optimizer
# Note that the loss here is a scalar because reduction="mean",
# therefore, after subtracting element-wise the dimensions
# predicted - actuals, every element of the resulting tensor is squared.
# Finally, the tensor is flatten, the elements are summed and the sum is
# divided by the number of elements in the tensor.
# That means for us that we are going to train based on the mean squared error
# (squared L2 norm).
criterion = nn.MSELoss()
lr = 0.01
momentum = 0.9
optimizer = optim.SGD(na.parameters(), lr=lr, momentum=momentum)

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
epochs = 1
batches_in_train_dt = len(train_dt)
batches_in_val_dt = len(val_dt)
# vector of loss in one iteration inside the second loop. That is the loss
# vector from one specific batch in one specific epoch.
this_loss_vector = []
# average loss for an specific batch
this_loss_vector_average = []

for epoch in range(epochs):  # no. of epochs
    running_loss = 0
    for i, data in enumerate(train_dt, 1):
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
        print(
            (
                "Loss in [epoch: {epoch}/{epochs}, training batch: "
                "{batch}/{batches}] is {thisloss}"
            ).format(
                epoch=epoch + 1,
                epochs=epochs,
                batch=i,
                batches=batches_in_train_dt,
                thisloss=this_loss,
            )
        )
        print(this_loss)
        this_loss_vector.append(this_loss)
        # compute *average* loss
        this_loss_vector_average.append(float(this_loss)/batch_size)
        running_loss += this_loss
        if debug:
            print("Running loss follows")
            print(running_loss)

    average_epoch_loss = running_loss / len(train_dt)
    average_epoch_loses.append(average_epoch_loss)
    print("[Epoch %d] loss: %.19f" % (epoch + 1, average_epoch_loss))

print("Done Training")
print("Saving model...")
model_name = "Neural_Anthropometer_Model_{}.pt".format(
    datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S'))
model_path = os.path.join(model_path, model_name)
torch.save(na.state_dict(), model_path)
print("Model saved to {}".format(model_path))
average_epoch_loses_filepath = os.path.join(
    rootDir, "log", "average_epoch_loses_{}.csv".format(1)
)
df = pd.DataFrame(data=average_epoch_loses)
df.to_csv(
    "average_epoch_loses.csv",
    index=False,
)
plt.plot(np.array(this_loss_vector_average), "r")
plt.title("Training Curve (batch_size={}, lr={})".format(batch_size, lr))
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()
