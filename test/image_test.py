import sys
import os
sys.path.append(os.path.abspath('..'))

from src.difference import interp_f_diff
from src.optimization import image_energy, image_gradient, EnergyFunction, diff_regularizer

from scipy.interpolate import LinearNDInterpolator
from scipy import optimize

import matplotlib.pyplot as plt

import numpy as np

def func(x,y):
    return np.sin(2*np.pi*x) + np.sin(2*np.pi*y)

def deformation(x, y, mu_x, mu_y, sigma):
    d = (x-mu_x)**2 + (y-mu_y)**2

    return (1.0/(4*np.pi*sigma**2))*np.exp(-d/(sigma**2))

N     = 10
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

U_nd = np.zeros((points_fine.shape))

F_diff_int = LinearNDInterpolator(points_fine, np.ravel(F_diff))

print("image energy I0 I0: ", image_energy(F_int,F_int,points_fine,U_nd))
print("image energy I0 I1: ", image_energy(F_int,F_diff_int,points_fine,U_nd))

E = image_energy(F_int,F_diff_int,points_fine,U_nd)

dx = 1e-3
grad = image_gradient(F_int,F_diff_int,points_fine,U_nd,dx)
grad_est = np.zeros((grad.shape))
for j in range(2):
    for i in range(U_nd.shape[0]):
        U_nd[i,j] += dx
        E_f = image_energy(F_int,F_diff_int,points_fine,U_nd)
        grad_est[i,j] = (E_f-E)/dx
        U_nd[:,:] = 0

##############################################
# Optimization
##############################################
regularizers = []
r1 = lambda x: 0.0000005*diff_regularizer(x, dx, -0.01, 0.1)
regularizers.append(r1)

U_exact = np.concatenate((U,U)).reshape((4*N*N,2))
print(r1(U_exact))

energy_function = EnergyFunction(F_int,F_diff_int,points_fine,dx,
    regularizers)

print(energy_function.energy(np.ravel(U_exact)))

U_vec = np.ravel(U_nd)

U_final = optimize.minimize(energy_function.energy, U_vec, method="BFGS",
    options={'disp':True}).x

U_final_nd = U_final.reshape((2*N,2*N,2))

X_final      = X_fine-U_final_nd[:,:,0]
Y_final      = Y_fine-U_final_nd[:,:,1]
Points_final = np.concatenate((np.ravel(X_final)[:,np.newaxis],
    np.ravel(Y_final)[:,np.newaxis]),axis=1)

F_final = F_int(Points_final).reshape((2*N,2*N))
##############################################
# Plot gradients
##############################################
f,axarr = plt.subplots(2,3)
grad = grad.reshape((2*N,2*N,2))
grad_est = grad_est.reshape((2*N,2*N,2))

p1 = axarr[0,0].imshow(grad[:,:,0], cmap='rainbow')
plt.colorbar(p1, ax=axarr[0,0])

p2 = axarr[0,1].imshow(grad_est[:,:,0], cmap='rainbow')
plt.colorbar(p2, ax=axarr[0,1])

p5 = axarr[0,2].imshow(np.abs(grad_est[:,:,0]-grad[:,:,0]), cmap='rainbow')
plt.colorbar(p5, ax=axarr[0,2])

p3 = axarr[1,0].imshow(grad[:,:,1], cmap='rainbow')
plt.colorbar(p3, ax=axarr[1,0])
# plt.colorbar(p3, ax=axarr[1,0])

p4 = axarr[1,1].imshow(grad_est[:,:,1], cmap='rainbow')
plt.colorbar(p4, ax=axarr[1,1])

p6 = axarr[1,2].imshow(np.abs(grad_est[:,:,1]-grad[:,:,1]), cmap='rainbow')
plt.colorbar(p6, ax=axarr[1,2])
plt.show()

##############################################
# Plot
##############################################
f,axarr = plt.subplots(3,2)
p1 = axarr[0,0].imshow(F, cmap='rainbow')
plt.colorbar(p1, ax=axarr[0,0])

p2 = axarr[0,1].imshow(F_fine, cmap='rainbow')
plt.colorbar(p2, ax=axarr[0,1])

p3 = axarr[1,0].quiver(U, -U)
# plt.colorbar(p3, ax=axarr[1,0])

p4 = axarr[1,1].imshow(F_diff, cmap='rainbow')
plt.colorbar(p4, ax=axarr[1,1])

p5 = axarr[2,0].quiver(U_final_nd[:,:,0], -U_final_nd[:,:,1])

p6 = axarr[2,1].imshow(F_final, cmap='rainbow')
plt.colorbar(p6, ax=axarr[2,1])
plt.show()
