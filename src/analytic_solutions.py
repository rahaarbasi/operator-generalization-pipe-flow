import numpy as np


def newtonian_pipe_velocity(r, radius=1.0, viscosity=1.0, dpdz=-1.0):
    """
    Analytical velocity profile for fully developed Newtonian pipe flow.

    Parameters
    ----------
    r : array-like
        Radial coordinate.
    radius : float
        Pipe radius.
    viscosity : float
        Dynamic viscosity.
    dpdz : float
        Axial pressure gradient, dp/dz. Usually negative for flow in +z direction.

    Returns
    -------
    u : array-like
        Axial velocity profile u_z(r).
    """
    r = np.asarray(r)
    return -(dpdz) / (4.0 * viscosity) * (radius**2 - r**2)


def newtonian_wall_shear_stress(radius=1.0, dpdz=-1.0):
    """
    Wall shear stress magnitude for fully developed Newtonian pipe flow.
    """
    return -0.5 * dpdz * radius


def newtonian_average_velocity(radius=1.0, viscosity=1.0, dpdz=-1.0):
    """
    Cross-sectional average velocity for fully developed Newtonian pipe flow.
    """
    return -(dpdz) * radius**2 / (8.0 * viscosity)