import numpy as np
from src.difference import interp_f_diff

def image_energy(I0, I1, X, U):
    """
    Image energy for deformation matching

    args:
        I0 - NDInterpolant of original image
        I1 - NDInterpolant of target image
        X  - Grid over target image (npoints, ndimensions)
        U  - current deformation (npoints,ndimensions)
    """

    Z = X-U
    I0_values = I0(Z)
    I1_values = I1(X)
    E = np.sum((I0_values - I1_values)**2)
    return E

def image_gradient(I0, I1, X, U, dx):
    Z = X-U

    I0_values = I0(Z)
    I1_values = I1(X)

    dimensions = U.shape[1]
    grad = np.zeros(U.shape):
    for i in range(dimensions):
        didx = interp_f_diff(I0,Z,dx,dimension=i)
        grad[:,i] = 2*(I0_values-I1_values)*didx*-1

    return grad
