from vtkUtil import mesh_to_polydata

class ImageModel(object):
    def __init__(self, bg_image, grid_points):
        self.bg_image    = bg_image
        self.grid_points = grid_points
        self.mesh        = None

    def set_mesh(self, mesh):
        self.mesh = mesh

    def get_binary_image(self):
        if self.mesh == None:
            raise RuntimeError("get_binary_image error, mesh==None")

        polydata = mesh_to_polydata(self.mesh)


    def update(self, displacement_interp):
        pass

    def __call__(self, points):
        return self.image_model_interp(points)
