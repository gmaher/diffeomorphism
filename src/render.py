import numpy as np
import trimesh

from meshrender import Scene, MaterialProperties, AmbientLight, PointLight, SceneObject, VirtualCamera

# Start with an empty scene
scene = Scene()


#create trimesh
verts = np.zeros((3,3))
verts[0] = np.array([1,0,0])
verts[1] = np.array([0,1,0])
verts[2] = np.array([0,0,1])

faces = np.zeros((1,3))
faces[0] = np.array([0,1,2])

mesh = trimesh.base.Trimesh(vertices=verts, faces=faces)
