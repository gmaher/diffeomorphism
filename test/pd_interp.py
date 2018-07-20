import sys
import os
sys.path.append(os.path.abspath('..'))

from src.optimization import CostFunction, diff_regularizer, image_energy
from src import mesh
from src import vtkUtil

from scipy.interpolate import LinearNDInterpolator
from scipy import optimize

import matplotlib.pyplot as plt

import numpy as np

def func(x,y):
    return np.sin(2*np.pi*x) + np.sin(2*np.pi*y)

def deformation(x, y, mu_x, mu_y, sigma):
    d = (x-mu_x)**2 + (y-mu_y)**2

    return (1.0/(4*np.pi*sigma**2))*np.exp(-d/(sigma**2))

def get_mesh(n=20,x=[0.3,0.45],r=0.2):

    verts = np.zeros((n,3))

    for i in range(n):
        x_i = np.cos(2*np.pi*i*1.0/n)*r+x[0]
        y_i = np.sin(2*np.pi*i*1.0/n)*r+x[1]

        verts[i] = [x_i,y_i,0]

    lines = np.zeros((n,2)).astype(int)

    for i in range(n-1):
        lines[i] = [i,i+1]
    lines[n-1] = [n-1,0]

    m = mesh.Mesh(verts,lines,np.array([]))
    return m

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
# Binary image construction
##############################################
MESH = get_mesh()
pd   = vtkUtil.mesh_to_polydata(MESH)

spacing    = np.array([1.0/Nfine, 1.0/Nfine, 1.0/Nfine])
dimensions = [Nfine,Nfine,1]
origin     = [0,0,0]

binary_image = vtkUtil.polydata_to_binary_image(pd, dimensions, spacing, origin)
binary_image_np =vtkUtil.vtk_image_to_numpy(binary_image)
binary_interp = LinearNDInterpolator(points_fine-spacing[:2]/2, np.ravel(binary_image_np))

binary = binary_interp(points_fine)
binary = binary.reshape((Nfine,Nfine))

M_in = np.nansum(binary*F_diff)*1.0/np.nansum(binary)
M_out = np.nansum((1-binary)*F_diff)*1.0/np.nansum(1-binary)
image_model = M_out + (M_in-M_out)*binary

##############################################
# optimization
##############################################
def image_cost_function(U, image_interp, binary_interp, X):
    Z = X+U
    I_bin = binary_interp(X)
    I     = image_interp(Z)

    M_in    = np.nansum(I_bin*I)*1.0/np.nansum(I_bin)
    M_out   = np.nansum((1-I_bin)*I)*1.0/np.nansum(1-I_bin)
    I_model = M_out + I_bin*(M_in-M_out)

    return np.nanmean((I-I_model)**2)

U0     = np.zeros(points_fine.shape)
U0_vec = np.ravel(U0)
U0_int = LinearNDInterpolator(points_fine, U0)

U_final = U0.copy()
U_final[:,1] -= 0.1

U_final_vec = np.ravel(U_final)
U_final_int = LinearNDInterpolator(points_fine,U_final)

F_final = F_int(Points_diff+U_final).reshape((Nfine,Nfine))

f1 = lambda U: image_cost_function(U, F_diff_int, binary_interp, points_fine)
f2 = lambda U: 0.05*diff_regularizer(U, points_fine, U_int, 1e-3)

cost_function = CostFunction(points_fine, U0_int, functions=[f1, f2])

print(cost_function(np.ravel(U0)))
print(cost_function(np.ravel(U_final)))

# U_final = optimize.minimize(cost_function, U0_vec, method="BFGS",
#     options={'disp':True}).x

bounds = [(-0.3,0.3)]*len(U0_vec)
U_final = optimize.differential_evolution(cost_function, bounds=bounds, maxiter=10,
    disp=True).x

#
U_final = U_final.reshape(U_fine.shape)

U_final_points = U_final.reshape((Nfine*Nfine,2))
U_final_interp = LinearNDInterpolator(points_fine,U_final_points)

Points_final = np.concatenate((np.ravel(X_fine+U_final[:,:,0])[:,np.newaxis],
    np.ravel(Y_fine+U_final[:,:,1])[:,np.newaxis]),axis=1)

F_final = F_int(Points_final).reshape((Nfine,Nfine))

verts_final = MESH.vertices.copy()
verts_final = verts_final[:,:2]+U_final_interp(verts_final[:,:2])

# ##############################################
# # Plot
# ##############################################

#image_model
plt.figure()
plt.imshow(binary, cmap='gray', extent=[0,1,1,0])
plt.plot(MESH.vertices[:,0],MESH.vertices[:,1], linewidth=2, color='r')
plt.colorbar()
plt.show()
plt.close()

plt.figure()
plt.imshow(image_model, cmap='jet', extent=[0,1,1,0])
plt.plot(MESH.vertices[:,0],MESH.vertices[:,1], linewidth=2, color='r')
plt.colorbar()
plt.show()
plt.close()

plt.figure()
plt.imshow(F_diff, cmap='jet', extent=[0,1,1,0])
plt.plot(MESH.vertices[:,0],MESH.vertices[:,1], linewidth=2, color='r')
plt.colorbar()
plt.show()
plt.close()

plt.figure()
plt.imshow(F_final, cmap='jet', extent=[0,1,1,0])
plt.plot(MESH.vertices[:,0],MESH.vertices[:,1], linewidth=2, color='r')
plt.colorbar()
plt.show()
plt.close()

plt.figure()
plt.imshow(F_diff, cmap='jet', extent=[0,1,1,0])
plt.plot(MESH.vertices[:,0],MESH.vertices[:,1], linewidth=2, color='b')
plt.plot(verts_final[:,0], verts_final[:,1], linewidth=2, color='k')
plt.colorbar()
plt.show()
plt.close()

# f,axarr = plt.subplots(5,2)
# p1 = axarr[0,0].imshow(U_coarse[:,:,0], cmap='rainbow')
# plt.colorbar(p1, ax=axarr[0,0])
#
# p2 = axarr[0,1].imshow(U_coarse[:,:,1], cmap='rainbow')
# plt.colorbar(p2, ax=axarr[0,1])
#
# p7 = axarr[1,0].imshow(U_final[:,:,0], cmap='rainbow')
# plt.colorbar(p7, ax=axarr[1,0])
#
# p8 = axarr[1,1].imshow(U_final[:,:,1], cmap='rainbow')
# plt.colorbar(p8, ax=axarr[1,1])
#
# p3 = axarr[2,0].imshow(U_fine[:,:,0], cmap='rainbow')
# plt.colorbar(p3, ax=axarr[2,0])
#
# p4 = axarr[2,1].imshow(U_fine[:,:,1], cmap='rainbow')
# plt.colorbar(p4, ax=axarr[2,1])
#
# p5 = axarr[3,0].imshow(F_fine, cmap="rainbow")
# plt.colorbar(p5, ax=axarr[3,0])
#
# p6 = axarr[3,1].imshow(F_diff, cmap='rainbow')
# plt.colorbar(p6, ax=axarr[3,1])
#
# p9 = axarr[4,0].imshow(F_final, cmap='rainbow')
# plt.colorbar(p9, ax=axarr[4,0])
#
# p10 = axarr[4,1].imshow(F_final, cmap='rainbow')
# plt.colorbar(p10, ax=axarr[4,1])
# plt.show()
