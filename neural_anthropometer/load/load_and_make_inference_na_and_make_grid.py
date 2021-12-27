# -*- coding: utf-8 -*-
"""
@author: neoglez
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt  # for plotting
from torch.utils.data import DataLoader
import os
import neural_anthropometer as na

rootDir = os.path.abspath(os.path.curdir)
root_dir = os.path.join(rootDir, "..", "..", "dataset")
model_path = os.path.join(rootDir, "..", "..", "model")
# If you want to make inference with a specific model, change following line
model_name = "Neural_Anthropometer_Model_20-12-2021_15-33-10_fold-0.pt"
model_path = os.path.join(model_path, model_name)

transform = na.TwoDToTensor()

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = False
# create a new model, initialize random parameters
model = na.NeuralAnthropometer(debug=False)  # .to(device)
model.load_state_dict(torch.load(model_path))
model.eval()
batch_size = 4

# loss and optimizer
# Note that the loss here is a scalar because reduction="mean",
# therefore, after subtracting element-wise the dimensions
# predicted - actuals, every element of the resulting tensor is squared.
# Finally, the tensor is flatten, the elements are summed and the sum is
# divided by the number of elements in the tensor.
# That means for us that we are going to test based on the mean squared error
# (squared L2 norm).
# Just as info
criterion = nn.MSELoss()

# 300 instances to test
na_test = na.NeuralAnthropometerSyntheticImagesDataset(
        root_dir=root_dir, transform=transform
    )

test_dataset = na_test

test_dt = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

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
# integer indices. Note that all HBD are given in
# meters, so if you want to convert them to cm, you have to multiply by 100.

for i, data in enumerate(test_dt, 1):
    # move to GPU if available
    actual_hbds = data["annotations"]["human_dimensions"]
    inputs = data["image"]

    predicted_hbds = model(inputs)
    images = data["image"]
    metadata = data["subject_string"]
    # create grid of images and annotations
    fig = na.image_grid(
        images,
        actual_hbds,
        predicted=predicted_hbds,
        subject_matada=metadata,
        background="white",
    )
    plt.show()
    break
