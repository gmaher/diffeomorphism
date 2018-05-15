import os
import sys
import numpy as np
sys.path.append(os.path.abspath('..'))

from src import mesh, vtkUtil

vertices = np.array([[1,0,0], [0,1,0], [0,0,1]])
lines    = np.array([[0,1],[1,2],[2,0]])
faces    = np.array([[0,1,2]])

m = mesh.Mesh(vertices, lines, faces)

f = lambda a: np.array([2*a[0],3*a[1],2*a[2]])

print m.vertices

m.apply(f)

print m.vertices

vtkUtil.saveMeshPolydata(m, '../data/mesh_test.vtk')
