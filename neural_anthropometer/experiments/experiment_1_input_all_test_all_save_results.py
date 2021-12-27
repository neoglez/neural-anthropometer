#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: neoglez
"""
import torch
import torch.nn as nn
import torch.optim as optim
import os
import neural_anthropometer as na
import numpy as np
import datetime
from sklearn.model_selection import KFold
import locale

locale.setlocale(locale.LC_NUMERIC, "C")

rootDir = os.path.join("..", "..", "dataset")
rootDir = os.path.abspath(rootDir)


model_path = os.path.join(rootDir, "..", "model")
kfold_results_path = os.path.join(rootDir, "..", "results")
kfold_results_path = os.path.join(kfold_results_path, "experiment1")
kfold_results_numpy_name = "kfold_results.npy"
results_file = os.path.join(kfold_results_path, kfold_results_numpy_name)

kfold_results_info_name = "results_info.svg"
kfold_results_info_name = "kfold_info.svg"

# Configuration options
k_folds = 5
batch_size = 100
num_epochs = 20
# CUDA for PyTorch
# Set fixed random number seed
torch.manual_seed(41)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = False

# For fold results
results = {}

# loss and optimizer
# Note that the loss here is a scalar because reduction="mean",
# therefore, after subtracting element-wise the dimensions
# predicted - actuals, every element of the resulting tensor is squared.
# Finally, the tensor is flatten, the elements are summed and the sum is
# divided by the number of elements in the tensor.
# That means for us that we are going to train based on the mean squared error
# (squared L2 norm).
transform = na.TwoDToTensor()
criterion = nn.MSELoss()
# performance_criterion = nn.MSELoss(reduction="none")
error_criterion = nn.MSELoss()

lr = 0.01
momentum = 0.9

dataset = na.NeuralAnthropometerSyntheticImagesDataset(
    root_dir=rootDir, transform=transform
)

# Define the K-fold Cross Validator
kfold = KFold(n_splits=k_folds, shuffle=True)

model_timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

# Here we collect the results: we are going to record the estimated and
# and actual measurements over the 5 folds. In this step, we do not
# calculate error, loss or any other statistic. We merely save the data to
# analyze it later. Since we have 5 folds, and 8 HBD estimated and actuals,
# we must have at the end a tensor of size 5 X 8 X 2.
results = np.zeros((k_folds,
                    int(len(dataset)/ k_folds),
                    2,
                    8))

# Start print
print("--------------------------------")

# K-fold Cross Validation model evaluation
for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
    # Print
    print(f"FOLD {fold}")
    print("--------------------------------")

    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

    # Define data loaders for training and testing data in this fold
    trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_subsampler
    )
    testloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, sampler=test_subsampler
    )

    # Init the neural network
    network = na.NeuralAnthropometer(debug=False).to(device)

    # Initialize optimizer
    optimizer = optim.SGD(network.parameters(), lr=lr, momentum=momentum)

    # Run the training loop for defined number of epochs
    for epoch in range(0, num_epochs):

        # Print epoch
        print(f"Starting epoch {epoch+1}")

        # Set current loss value
        current_loss = 0.0

        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader, 0):
            # Get inputs
            # move to GPU if available
            targets = data["annotations"]["human_dimensions"].to(device)
            inputs = data["image"].to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            outputs = network(inputs)

            # Compute loss
            loss = criterion(outputs, targets)

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()

            # Print statistics
            current_loss += loss.item()
            if i % 25 == 24:
                print(
                    "Loss after mini-batch %5d: %.5f"
                    % (i + 1, current_loss / 25)
                )
                current_loss = 0.0

    # Process is complete.
    print("Training process has finished. Saving trained model.")

    # Saving the model
    model_name = (
        "Neural_Anthropometer_Model_"
        + model_timestamp
        + "_fold-{}.pt".format(fold)
    )
    model_file = os.path.join(model_path, model_name)
    torch.save(network.state_dict(), model_file)

    # Evaluation for this fold
    kfold_this_error = 0
    with torch.no_grad():
        # Iterate over the test data and generate predictions
        for i, data in enumerate(testloader, 0):

            # Get inputs
            targets = data["annotations"]["human_dimensions"].to(device)
            inputs = data["image"].to(device)

            # Generate outputs
            outputs = network(inputs)

            # record for this fold estimated and actuals
            # first the estimated
            results[fold][i][0] = outputs.detach().cpu().numpy().T.flatten()
            # second the actuals
            results[fold][i][1] = targets.detach().cpu().numpy().T.flatten()

        # Average error for this fold
        kfold_performace_error = kfold_this_error / len(testloader)
        # Print accuracy. In our case MSE over all dimensions.
        # The smaller, the better. Note that we have to average the MSE across
        # the mini-batches.
        print("Estimated and actual recorded for fold {:d}".format(fold))
        print("--------------------------------")

# Print fold results
print(f"K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS RECORDED")
print("--------------------------------")

# save results in numpy format
with open(results_file, "wb") as f:
    np.save(f, results)
