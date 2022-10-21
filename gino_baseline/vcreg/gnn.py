from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geom_nn

class GNN(nn.Module):
    """implements a graphical neural network in pytorch. In particular, we will use pytorch geometric's nn_conv module so we can apply a neural network to the edges.
    """

    def __init__(
        self,
        n_hidden=64,
        num_convolution_blocks=2,
        n_output=1,
        supervised=False,
        n_node_features = 5,
        n_edge_features = 4
    ) -> None:
        """create the gnn

        Args:
            n_node_features: input features on each node
            n_edge_features: input features on each edge
            n_hidden: hidden features within the neural networks (embeddings, nodes after graph convolutions, etc.)
            n_output: how many output features
            num_convolution_blocks: how many blocks convolutions should be performed. A block may include multiple convolutions
        """
        super(GNN, self).__init__()

        self.supervised=supervised

        self.embedding = nn.Linear(n_node_features, n_hidden)
        self.conv_blocks = nn.ModuleList()

        for block_idx in range(num_convolution_blocks):
            self.conv_blocks.append(geom_nn.RGCNConv(n_hidden, n_hidden, n_edge_features, aggr="mean"))
            self.conv_blocks.append(geom_nn.MFConv(n_hidden, n_hidden, aggr="mean"))

        self.linear_1 = nn.Linear(n_hidden, n_hidden)
        self.linear_2 = nn.Linear(n_hidden, n_output)

    def forward(
        self,
        batch
    ) -> torch.Tensor:
        """
        Args:
            x: Input features per node
            edge_index: List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
            edge_attr: edge attributes (pytorch geometric notation)
            batch_idx: Index of batch element for each node

        Returns:
            prediction
        """

        x = batch.x[:, :5]

        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        batch_idx = batch.batch

        x = self.embedding(x)
        edge_type = torch.argmax(edge_attr, dim=1)


        for idx, module in enumerate(self.conv_blocks):
            if idx != 0:
                x = F.relu(x)
            
            if idx % 2 == 0:
                x = module(x, edge_index, edge_type)
                x = F.relu(x)
            else:
                x = module(x, edge_index)
        
        x = geom_nn.global_mean_pool(x, batch_idx)

        if self.supervised:
            x = F.relu(self.linear_1(x))
            x = self.linear_2(x)

        return x

    @property
    def device(self):
        """
        Returns the device on which the model is. Can be useful in some situations.
        """
        return next(self.parameters()).device
