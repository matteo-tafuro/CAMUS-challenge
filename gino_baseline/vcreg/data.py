
from __future__ import absolute_import, division, print_function

from warnings import warn

import numpy as np

import torch
from torch_geometric.data.batch import Batch
from torch_geometric.data.dataset import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import QM9
from torch_geometric.utils import to_dense_adj


# Constants for the QM9 problem
MAX_NUM_NODES = 29  # The larget molecule has 29 atoms
Z_ONE_HOT_DIM = 5  # The one hot encoding of the element (z) has 5 unique values
EDGE_ATTR_DIM = 4  # The one hot encoding of the edges have 4 unique values
LABEL_INDEX = 7  # The energy of atomization at 0K exists at index 7
FLAT_INPUT_DIM = 3509  # The largest molecule in QM9 with all the node features and edges flattened is this long


def get_node_features(molecules: Batch) -> torch.Tensor:
    """return the node features permitted for the problem. the features are one hot encodings of the atomic number.

    Args:
        molecules: pytorch geometric batch of molecules

    Returns:
        z: a one hot tensor based on the node's atomic number. (batch, Z_ONE_HOT_DIM)
    """
    return molecules.x[:, :Z_ONE_HOT_DIM]


def get_labels(molecules: Batch, LABEL_INDEX=LABEL_INDEX) -> torch.Tensor:
    """return the labels for our problem. the labels are u0.

    Args:
        molecules: pytorch geometric batch of molecules

    Returns:
        u0 labels: a tensor of labels (batch, 1)
    """
    return molecules.y[:, LABEL_INDEX]


def get_z_one_hot(molecules: Batch) -> torch.Tensor:
    """accesses the node features from batch of molecules and converts them into a dense, padded tensor.

    Args:
        molecules: a batch of molecules from pytorch geometric

    Returns:
        dense one hot vector representing atomic type
    """
    batch_size = molecules.batch.unique().numel()
    one_hots = get_node_features(molecules)
    _, counts = molecules.batch.unique(return_counts=True)
    row_position = torch.cat(
        [torch.arange(c, device=molecules.batch.device) for c in counts]
    )
    indices = torch.stack([molecules.batch, row_position])
    z_one_hot = torch.sparse_coo_tensor(
        indices=indices,
        values=one_hots,
        size=(batch_size, MAX_NUM_NODES, Z_ONE_HOT_DIM),
    ).to_dense()
    return z_one_hot


def get_dense_adj(molecules: Batch) -> torch.Tensor:
    """accesses the edge index and attr from a batch of molecules and converts them into a dense, padded adjacency matrix.

    Args:
        molecules: pytorch geometric batch

    Returns:
        dense, padded adjacency matrix with edge attr
    """
    dense_adj = to_dense_adj(
        molecules.edge_index, molecules.batch, molecules.edge_attr, MAX_NUM_NODES
    )
    return dense_adj


def get_mlp_features(molecules: Batch) -> torch.Tensor:
    """accesses the batch and produces a padded, flattened tensor suitable for an mlp.

        molecules: pytorch geometric batch

    Returns:
        dense, padded, flattened input features
    """
    z_one_hot = get_z_one_hot(molecules)
    dense_adj = get_dense_adj(molecules)
    x = torch.cat([z_one_hot.flatten(1), dense_adj.flatten(1)], dim=1)
    return x


def process_qm9(dataset=None, device="cuda") -> 'tuple[Dataset, Dataset, Dataset]':
    """Download the QM9 dataset from pytorch geometric. Put it onto the device. Split it up into train / validation / test.

    Args:
        data_dir: the directory to store the data.
        device: put the data onto this device.

    Returns:
        train dataset, validation dataset, test dataset.
    """
    if dataset is None:
        dataset = QM9('./QM9')

    dataset = dataset.shuffle(return_perm=False)

    # z score / standard score targets to mean = 0 and std = 1.
    mean = dataset.data.y.mean(dim=0, keepdim=True)
    std = dataset.data.y.std(dim=0, keepdim=True)
    dataset.data.y = (dataset.data.y - mean) / std
    mean, std = mean[:, LABEL_INDEX].item(), std[:, LABEL_INDEX].item()

    # Move the data to the device (it should fit on lisa gpus)
    dataset.data = dataset.data.to(device)

    len_train = 100_000
    len_val = 10_000

    train = dataset[:len_train + len_val]
    test = dataset[len_train + len_val : ]

    assert len(dataset) == len(train) + len(test)

    return train, test



def get_loaders(
    dataset=None, data_dir=None, batch_size=128, type="vicreg", LABEL_INDEX=7
):
    """a full training cycle of an mlp / gnn on qm9.

    Args:
        model: a differentiable pytorch module which estimates the U0 quantity
        lr: learning rate of optimizer
        batch_size: batch size of molecules
        epochs: number of epochs to optimize over
        seed: random seed
        data_dir: where to place the qm9 data

    Returns:
        model: the trained model which performed best on the validation set
        test_loss: the loss over the test set
        permuted_test_loss: the loss over the test set where atomic indices have been permuted
        val_losses: the losses over the validation set at every epoch
        logging_info: general object with information for making plots or whatever you'd like to do with it

    """
    # Loading the dataset
    if dataset is None:
        dataset = QM9(data_dir)

    dataset = process_qm9(dataset)

    qm9_train, qm9_test = dataset

    if type == "supervised":
        qm9_train_val = qm9_train
        
        qm9_train = qm9_train_val[0:-len(qm9_test)]
        qm9_val = qm9_train_val[-len(qm9_test) :]

    train_loader = DataLoader(
        qm9_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        exclude_keys=["idx", "z", "name"],
    )
    _train_loader = DataLoader(
        qm9_train if type == "vicreg" else qm9_val,
        batch_size=batch_size,
        drop_last=True,
        exclude_keys=["idx", "z", "name"],
    )
    test_loader = DataLoader(
        qm9_test, batch_size=batch_size, exclude_keys=["idx", "z", "name"],
        drop_last=True
    )

    return train_loader, _train_loader, test_loader
