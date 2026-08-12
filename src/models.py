"""
Neural network models for pipe-flow surrogate modeling.

This module currently defines a simple multilayer perceptron (MLP) that learns
a parametric mapping from pipe-flow inputs to the axial velocity.

Current supervised learning task:

    (r, R, mu, dp/dz) -> u(r)

where:
    r      : radial coordinate
    R      : pipe radius
    mu     : dynamic viscosity
    dp/dz  : imposed axial pressure gradient
    u(r)   : axial velocity at radial location r

This model is the first data-driven baseline before adding physics-informed
loss terms or neural-operator architectures.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Simple multilayer perceptron for Newtonian pipe-flow surrogate modeling.

    The model takes four scalar inputs:

        [r, radius, viscosity, dpdz]

    and predicts one scalar output:

        [u]

    This is a data-driven surrogate model. It does not yet enforce the
    governing equation or boundary conditions directly.
    """

    def __init__(self, input_dim=4, hidden_dim=64, output_dim=1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch_size, 4).

        Returns
        -------
        torch.Tensor
            Predicted axial velocity with shape (batch_size, 1).
        """
        return self.net(x)