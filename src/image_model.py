from vtkUtil import mesh_to_polydata, vtk_image_to_numpy, polydata_to_binary_image
from scipy.interpolate import LinearNDInterpolator

class ImageModel(object):
    def __init__(self, bg_image, grid_points):
        self.bg_image    = bg_image
        self.grid_points = grid_points
        self.mesh        = None
        self.dimensions  = self.bg_image.shape

    def set_spacing(self,spacing):
        self.spacing = spacing

    def set_origin(self,origin):
        self.origin = origin

    def set_mesh(self, mesh):
        self.mesh = mesh

    def get_binary_image(self):
        if self.mesh == None:
            raise RuntimeError("get_binary_image error, mesh==None")

        polydata = mesh_to_polydata(self.mesh)

        binary_vtk = polydata_to_binary_image(polydata, self.dimension,
            self.spacing, self.origin)

        self.binary_image  = vtk_image_to_numpy(binary_vtk)

        return self.binary_image

    def interpolate(self):
        self.binary_interp = LinearNDInterpolator(self.grid_points,
            np.ravel(self.binary_image))

    def __call__(self, points):
        return self.binary_interp(points)
