import torch
import torch.nn as nn
import torch.nn.functional as F

from gnn import GNN

def exclude_bias_and_norm(p):
    return p.ndim == 1

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def Projector(dims):
    layers = []

    for i in range(len(dims) - 2):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(nn.BatchNorm1d(dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(dims[-2], dims[-1], bias=False))
    return nn.Sequential(*layers)

class VICReg(nn.Module):
    def __init__(self, encoder,
        projector_dims=(4096, 4096, 4096, 8192), siamese=True,
        global_inv_coeff=25.0, std_coeff=25.0, cov_coeff=1.0, exp_coeff=4.0,

    ):

        super().__init__()

        self.num_features = projector_dims[-1]

        self.exp_coeff = exp_coeff
        self.global_inv_coeff = global_inv_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

        self.encoder = encoder

        if siamese:
            self._encoder = self.encoder
        else:
            self._encoder = self.encoder

        self.projector = Projector(projector_dims)

    def forward(self, batch):

        batch = batch.to(self.device)

        batch, _batch = batch[:, 0, :, :].unsqueeze(1), batch[:, 1, :, :].unsqueeze(0)

        x = self.projector(self.encoder(batch))
        y = self.projector(self._encoder(_batch))

        inv_loss = F.mse_loss(x, y)

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        batch_size = x.shape[0]

        cov_x = (x.T @ x) / (batch_size - 1)
        cov_y = (y.T @ y) / (batch_size - 1)

        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        loss = (
            self.global_inv_coeff * inv_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )

        return loss

    @property
    def device(self):
        return next(self.parameters()).device


def train_vicreg(
    model,
    loader,
    optimizer,
    epochs,
    device="cuda",
    log_every_n_batches=5
):
    losses = []

    model.to(device)

    for epoch in range(epochs):
        print('Starting epoch {}'.format(epoch))

        model.train()

        for batch in range(len(loader)):
            optimizer.zero_grad()

            batch = next(iter(loader))
            batch.to(device)

            loss = model(batch)

            loss.backward()
            optimizer.step()

            if batch % log_every_n_batches == 0:
                print('Batch {} loss: {}'.format(batch, loss.item()))

            losses.append(loss.item())
    
    return model, losses
