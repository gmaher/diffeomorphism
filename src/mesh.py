import numpy as np

class Mesh(object):
    def __init__(self, vertices, lines, faces=None):
        self.vertices = vertices
        self.lines    = lines
        self.faces    = faces

    def pushVertices(self, vertices):
        if not len(vertices.shape) == 4:
            raise RuntimeError("Pushed vertices has shape {} must have len 4"
                .format(vertices.shape))

        self.vertices = np.concatenate((self.vertices,vertices))

    def pushLines(self, lines):
        if not len(lines.shape) == 3:
            raise RuntimeError("Pushed lines has shape {} must have len 3"
                .format(lines.shape))

        self.lines = np.concatenate((self.lines,lines))

    def pushFaces(self, faces):
        """
        Faces are triangles
        """
        if not len(faces.shape) == 4:
            raise RuntimeError("Pushed faces has shape {} must have len 4"
                .format(lines.shape))

        if self.faces == None:
            self.faces = faces
        else:
            self.faces = np.concatenate((self.faces, faces))

    def apply(self, f):
        self.vertices = np.apply_along_axis(func1d=f,
            axis=0, self.vertices)
