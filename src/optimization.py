import numpy as np
from src.difference import interp_c_diff_2

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

def diff_regularizer(U, X, Uint, dx, alpha=0.01, gamma=1):
    N, d = U.shape

    R = np.zeros(U.shape)

    for i in range(d):
        R += -alpha*interp_c_diff_2(Uint,X,dx, dimension=d)

    R += gamma*U

    return np.nansum(R**2)

class EnergyFunction(object):
    def __init__(self, I0, I1, X, Uint, dx, regularizers=[]):
        self.I0 = I0
        self.I1 = I1
        self.X  = X
        self.Uint = Uint
        self.dx = dx
        self.regularizers = regularizers

    def energy(self,U):
        E  = 0
        Ur = U.reshape((self.X.shape))

        self.Uint.values = Ur

        for reg in self.regularizers:
            E += reg(Ur)

        E += image_energy(self.I0, self.I1, self.X, Ur)
        return E
