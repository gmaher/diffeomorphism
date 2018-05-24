import sys
import os
sys.path.append(os.path.abspath('..'))

from src.difference import interp_f_diff
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt

import numpy as np

def func(x,y):
    return np.sin(2*np.pi*x) + np.sin(2*np.pi*y)

def deformation(x, y, mu_x, mu_y, sigma):
    d = (x-mu_x)**2 + (y-mu_y)**2

    return (1.0/(4*np.pi*sigma**2))*np.exp(-d/(sigma**2))

N     = 25
MU    = [0.75,0.5]
SIGMA = 0.25
SCALE = 0.2
x = np.linspace(0,1,N)
x_fine = np.linspace(0,1,2*N)

X,Y = np.meshgrid(x,x)

X_fine, Y_fine = np.meshgrid(x_fine,x_fine)

F   = func(X,Y)

points = np.concatenate((np.ravel(X)[:,np.newaxis],
    np.ravel(Y)[:,np.newaxis]),axis=1)

values = np.ravel(F)

F_int = LinearNDInterpolator(points, values)

points_fine = np.concatenate((np.ravel(X_fine)[:,np.newaxis],
    np.ravel(Y_fine)[:,np.newaxis]),axis=1)

F_fine = F_int(points_fine).reshape((2*N,2*N))

U = SCALE*deformation(X_fine,Y_fine, MU[0], MU[1], SIGMA)

X_diff      = X_fine-U
Y_diff      = Y_fine-U
Points_diff = np.concatenate((np.ravel(X_diff)[:,np.newaxis],
    np.ravel(Y_diff)[:,np.newaxis]),axis=1)

F_diff = F_int(Points_diff).reshape((2*N,2*N))

f,axarr = plt.subplots(2,2)
p1 = axarr[0,0].imshow(F, cmap='rainbow')
plt.colorbar(p1, ax=axarr[0,0])

p2 = axarr[0,1].imshow(F_fine, cmap='rainbow')
plt.colorbar(p2, ax=axarr[0,1])

p3 = axarr[1,0].quiver(U, -U)
# plt.colorbar(p3, ax=axarr[1,0])

p4 = axarr[1,1].imshow(F_diff, cmap='rainbow')
plt.colorbar(p4, ax=axarr[1,1])
plt.show()
