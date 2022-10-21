from __future__ import absolute_import, division, print_function

import argparse
from copy import deepcopy
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data.batch import Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from data import *


def evaluate_model(
    model: nn.Module, data_loader: DataLoader, criterion: Callable, label_index=7,
tprint=False) -> float:
    """
    Performs the evaluation of the model on a given dataset.

    Args:
        model: trainable network
        data_loader: The data loader of the dataset to evaluate.
        criterion: loss module, i.e. torch.nn.MSELoss()
        permute: whether to permute the atoms within a molecule
    Returns:
        avg_loss: scalar float, the average loss of the model on the dataset.

    Hint: make sure to return the average loss of the whole dataset, 
          independent of batch sizes (not all batches might be the same size).
    """

    with torch.no_grad():
        loss = 0.0
        num_examples = 0
        for molecules in data_loader:

            target = molecules.y[:, label_index].unsqueeze(1)
            output = model(molecules)
            
            loss += F.mse_loss(output, target, reduction='sum').item()
            num_examples += len(molecules)
    
    avg_loss = loss / num_examples

    return avg_loss

def train_supervised(
    train_loader, val_loader, test_loader,
    model: nn.Module, lr: float, epochs: int, 
    label_index=7, optimizer=None, device='cuda',
    log_every_n_batches=100
):
    """
    A full training cycle of a gnn on qm9.
    """
    criterion = torch.nn.MSELoss()
    
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    best_model = None

    for epoch in range(epochs):

        print('Starting epoch {}'.format(epoch))

        model.train()

        #train_loss = 0.0

        for idx, molecules in enumerate(train_loader):
            molecules.to(device)

            #print molecules device
            optimizer.zero_grad()

            target = molecules.y[:, label_index].unsqueeze(1)
            output = model(molecules)
            
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            if idx % log_every_n_batches == 0:
                print('Batch {} loss: {}'.format(idx, loss.item()))

            #train_loss += loss.item()

        #train_loss /= len(train_dataloader.dataset)
        #print('Train loss at end of epoch: {}'.format(train_loss))

        #train_losses.append(train_loss)

        model.eval()

        val_loss = evaluate_model(model, val_loader, criterion, label_index=label_index)
        print('Val loss at end of epoch: {}'.format(val_loss))
        
        val_losses.append(val_loss)

        if best_model is None or val_loss < max(val_losses):
            best_model = deepcopy(model)


    test_loss = evaluate_model(best_model, test_loader, criterion, label_index=label_index, tprint=True)

    print('Test loss: {}'.format(test_loss))
    logging_info = {'train_losses': train_losses}

    return model, test_loss, val_losses, logging_info