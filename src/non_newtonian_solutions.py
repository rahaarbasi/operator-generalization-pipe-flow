"""
Analytical solutions for non-Newtonian pipe-flow benchmarks.

This module starts with power-law fluid pipe flow as a simple
non-Newtonian extension of the Newtonian Poiseuille-flow benchmark.
"""

import numpy as np


def power_law_pipe_velocity(r, radius=1.0, consistency=1.0, flow_index=0.8, dpdz=-1.0):
    """
    Analytical velocity profile for fully developed power-law pipe flow.

    Parameters
    ----------
    r : float or array-like
        Radial coordinate.
    radius : float
        Pipe radius.
    consistency : float
        Power-law consistency index K.
    flow_index : float
        Power-law flow behavior index n.
        n < 1: shear-thinning fluid
        n = 1: Newtonian-like behavior
        n > 1: shear-thickening fluid
    dpdz : float
        Axial pressure gradient.

    Returns
    -------
    numpy.ndarray
        Axial velocity profile.
    """
    r = np.asarray(r)

    pressure_force = -dpdz / 2.0

    coefficient = flow_index / (flow_index + 1.0)
    scale = (pressure_force / consistency) ** (1.0 / flow_index)

    return coefficient * scale * (
        radius ** ((flow_index + 1.0) / flow_index)
        - r ** ((flow_index + 1.0) / flow_index)
    )