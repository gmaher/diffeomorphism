import numpy as np
from src.difference import interp_f_diff, c_diff_2

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
    E = np.nansum((I0_values - I1_values)**2)
    return E

def image_gradient(I0, I1, X, U, dx):
    Z = X-U

    I0_values = I0(Z)
    I1_values = I1(X)

    dimensions = U.shape[1]
    grad = np.zeros(U.shape)
    for i in range(dimensions):
        didx = interp_f_diff(I0,Z,dx,dimension=i)
        grad[:,i] = 2*(I0_values-I1_values)*didx*-1

    return grad

def diff_regularizer(U, dx, alpha=0.1, gamma=1):
    N, d = U.shape
    u = U.reshape((int(N**0.5),int(N**0.5),d))

    R = np.zeros(u.shape)

    for i in range(d):
        R[:,:,i] += -alpha*(c_diff_2(u[:,:,i],dx, axis=0)+\
            c_diff_2(u[:,:,i],dx,axis=1))
        R[:,:,i] += gamma*u[:,:,i]

    return np.nansum(R**2)

class EnergyFunction(object):
    def __init__(self, I0, I1, X, dx, regularizers=[]):
        self.I0 = I0
        self.I1 = I1
        self.X  = X
        self.dx = dx
        self.regularizers = regularizers

    def energy(self,U):
        E  = 0
        Ur = U.reshape((self.X.shape))

        for reg in self.regularizers:
            E += reg(Ur)

        E += image_energy(self.I0, self.I1, self.X, Ur)
        return E
