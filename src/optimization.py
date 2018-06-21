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

    Z = X+U
    I0_values = I0(Z)
    I1_values = I1(X)
    E = np.nansum((I0_values - I1_values)**2)
    return E

def diff_regularizer(U, X, Uint, dx, alpha=-0.01, gamma=1):
    N, d = U.shape

    R = np.zeros(U.shape)

    for i in range(d):
        R += alpha*interp_c_diff_2(Uint,X,dx, dimension=i)

    R += gamma*U

    return np.nansum(R**2)

class CostFunction(object):
    def __init__(self, X, Uint, functions=[]):
        self.X    = X
        self.Uint = Uint
        self.functions = functions

    def __call__(self,U):
        Ur = U.reshape((self.X.shape))

        self.Uint.values = Ur
        E = 0
        for f in self.functions:
            E += f(Ur)

        return E
