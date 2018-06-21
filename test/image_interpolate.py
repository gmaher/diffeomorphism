import sys
import os
sys.path.append(os.path.abspath('..'))

from src.optimization import CostFunction, diff_regularizer, image_energy
from src import mesh

from scipy.interpolate import LinearNDInterpolator
from scipy import optimize

import matplotlib.pyplot as plt

import numpy as np

def func(x,y):
    return np.sin(2*np.pi*x) + np.sin(2*np.pi*y)

def deformation(x, y, mu_x, mu_y, sigma):
    d = (x-mu_x)**2 + (y-mu_y)**2

    return (1.0/(4*np.pi*sigma**2))*np.exp(-d/(sigma**2))

def get_mesh():
    verts = np.zeros((4,2))
    verts[0] = [0.2,0.2]
    verts[1] = [0.5,0.5]
    verts[2] = [0.3,0.75]
    verts[3] = [0.2,0.5]

    lines = np.zeros((4,2))
    lines[0] = [0,1]
    lines[1] = [1,2]
    lines[2] = [2,3]
    lines[3] = [3,0]

    m = mesh.Mesh(verts,lines,[])
    return m

MESH = get_mesh()

N     = 10
Nfine  = 2*N
MU    = [0.75,0.5]
SIGMA = 0.25
SCALE = 0.2
x = np.linspace(0,1,N)
x_fine = np.linspace(0,1,Nfine)

X,Y = np.meshgrid(x,x)

X_fine, Y_fine = np.meshgrid(x_fine,x_fine)

F   = func(X,Y)

points = np.concatenate((np.ravel(X)[:,np.newaxis],
    np.ravel(Y)[:,np.newaxis]),axis=1)

values = np.ravel(F)

F_int = LinearNDInterpolator(points, values)

points_fine = np.concatenate((np.ravel(X_fine)[:,np.newaxis],
    np.ravel(Y_fine)[:,np.newaxis]),axis=1)

F_fine = F_int(points_fine).reshape((Nfine,Nfine))

Ux = -SCALE*deformation(X_fine,Y_fine, MU[0], MU[1], SIGMA)
Uy = Ux[:,:]
U = np.concatenate((np.ravel(Ux)[:,np.newaxis], np.ravel(Uy)[:,np.newaxis]),
    axis=1)

X_diff      = X_fine+Ux
Y_diff      = Y_fine+Uy

Points_diff = np.concatenate((np.ravel(X_diff)[:,np.newaxis],
    np.ravel(Y_diff)[:,np.newaxis]),axis=1)

U_int  = LinearNDInterpolator(points_fine, U)

U_coarse = U_int(points).reshape((N,N,2))

U_fine   = U_int(points_fine).reshape((Nfine,Nfine,2))

F_diff = F_int(Points_diff).reshape((Nfine,Nfine))
F_diff_int = LinearNDInterpolator(points_fine, F_int(Points_diff))

##############################################
# optimization
##############################################
U0     = np.zeros(points_fine.shape)
U0_vec = np.ravel(U0)
U0_int = LinearNDInterpolator(points_fine, U0)

Ureg = U_int(points_fine)

reg0 = diff_regularizer(U0, points_fine, U0_int, 1e-3)
regU = diff_regularizer(Ureg, points_fine, U_int, 1e-3)

f1 = lambda U: image_energy(F_int, F_diff_int, points_fine, U)
f2 = lambda U: 0.001*diff_regularizer(U, points_fine, U_int, 1e-3)

cost_function = CostFunction(points_fine, U_int, functions=[f1,f2])

print(cost_function(np.ravel(U0)))
print(cost_function(np.ravel(Ureg)))

U_final = optimize.minimize(cost_function, U0, method="BFGS",
    options={'disp':True}).x

U_final = U_final.reshape(U_fine.shape)

Points_final = np.concatenate((np.ravel(X_fine+U_final[:,:,0])[:,np.newaxis],
    np.ravel(Y_fine+U_final[:,:,1])[:,np.newaxis]),axis=1)

F_final = F_int(Points_final).reshape((Nfine,Nfine))
##############################################
# Plot
##############################################
f,axarr = plt.subplots(5,2)
p1 = axarr[0,0].imshow(U_coarse[:,:,0], cmap='rainbow')
plt.colorbar(p1, ax=axarr[0,0])

p2 = axarr[0,1].imshow(U_coarse[:,:,1], cmap='rainbow')
plt.colorbar(p2, ax=axarr[0,1])

p7 = axarr[1,0].imshow(U_final[:,:,0], cmap='rainbow')
plt.colorbar(p7, ax=axarr[1,0])

p8 = axarr[1,1].imshow(U_final[:,:,1], cmap='rainbow')
plt.colorbar(p8, ax=axarr[1,1])

p3 = axarr[2,0].imshow(U_fine[:,:,0], cmap='rainbow')
plt.colorbar(p3, ax=axarr[2,0])

p4 = axarr[2,1].imshow(U_fine[:,:,1], cmap='rainbow')
plt.colorbar(p4, ax=axarr[2,1])

p5 = axarr[3,0].imshow(F_fine, cmap="rainbow")
plt.colorbar(p5, ax=axarr[3,0])

p6 = axarr[3,1].imshow(F_diff, cmap='rainbow')
plt.colorbar(p6, ax=axarr[3,1])

p9 = axarr[4,0].imshow(F_final, cmap='rainbow')
plt.colorbar(p9, ax=axarr[4,0])

p10 = axarr[4,1].imshow(F_final, cmap='rainbow')
plt.colorbar(p10, ax=axarr[4,1])
plt.show()
