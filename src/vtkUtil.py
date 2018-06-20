import vtk
import numpy as np
from vtk.util import numpy_support

def vtk_image_to_numpy(im):

    H,W,D = im.GetDimensions()
    sc = im.GetPointData().GetScalars()
    a = numpy_support.vtk_to_numpy(sc)
    a = a.reshape(H, W, D)

    assert a.shape==im.GetDimensions()
    return a

def read_mha(img_fn):
    reader = vtk.vtkMetaImageReader()

    reader.SetFileName(img_fn)
    reader.Update()
    return reader.GetOutput()

def readVTKPD(fn):
	'''
	reads a vtk polydata object from a file
	'''
	pd_reader = vtk.vtkXMLPolyDataReader()
	pd_reader.SetFileName(fn)
	pd_reader.Update()
	pd = pd_reader.GetOutput()
	return pd

def VTKSPtoNumpy(vol):
    '''
    Utility function to convert a VTK structured points (SP) object to a numpy array
    the exporting is done via the vtkImageExport object which copies the data
    from the supplied SP object into an empty pointer or array

    C/C++ can interpret a python string as a pointer/array

    This function was shamelessly copied from
    http://public.kitware.com/pipermail/vtkusers/2002-September/013412.html
    args:
    	@a vol: vtk.vtkStructuredPoints object
    '''
    exporter = vtkImageExport()
    exporter.SetInputData(vol)
    dims = exporter.GetDataDimensions()
    if np.sum(dims) == 0:
        return np.zeros((1,64,64))
    if (exporter.GetDataScalarType() == 3):
    	dtype = UnsignedInt8
    if (exporter.GetDataScalarType() == 4):
    	dtype = np.short
    if (exporter.GetDataScalarType() == 5):
    	dtype = np.int16
    if (exporter.GetDataScalarType() == 10):
    	dtype = np.float32
    if (exporter.GetDataScalarType() == 11):
    	dtype = np.float64
    a = np.zeros(reduce(np.multiply,dims),dtype)
    s = a.tostring()
    exporter.SetExportVoidPointer(s)
    exporter.Export()
    a = np.reshape(np.fromstring(s,dtype),(dims[2],dims[0],dims[1]))
    return a[0]

def VTKNumpytoSP(img_):
    img = img_.T

    H,W = img.shape

    sp = vtk.vtkStructuredPoints()
    sp.SetDimensions(H,W,1)
    sp.AllocateScalars(10,1)
    for i in range(H):
        for j in range(W):
            v = img[i,j]
            sp.SetScalarComponentFromFloat(i,j,0,0,v)

    return sp

def mesh_to_polydata(mesh):
    points = vtk.vtkPoints()
    lines  = vtk.vtkCellArray()
    faces  = vtk.vtkCellArray()

    v = mesh.vertices
    l = mesh.lines
    f = mesh.faces

    for i in range(v.shape[0]):
        points.InsertNextPoint(v[i][0],v[i][1],v[i][2])

    for i in range(l.shape[0]):
        vtkLine = vtk.vtkLine()
        vtkLine.GetPointIds().SetId(0,l[i][0])
        vtkLine.GetPointIds().SetId(1,l[i][1])
        lines.InsertNextCell(vtkLine)

    for i in range(f.shape[0]):
        vtkTri = vtk.vtkTriangle()
        vtkTri.GetPointIds().SetId(0,f[i][0])
        vtkTri.GetPointIds().SetId(1,f[i][1])
        vtkTri.GetPointIds().SetId(2,f[i][2])
        faces.InsertNextCell(vtkTri)

    pd = vtk.vtkPolyData()

    pd.SetPoints(points)
    pd.SetLines(lines)
    pd.SetPolys(faces)

    return pd

def save_mesh_polydata(mesh, filepath):
    pd = mesh2polydata(mesh)

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(filepath)
    writer.SetInputData(pd)
    writer.SetFileTypeToASCII()
    writer.Write()

def polydata_to_binary_image(polydata, dimensions, spacing, origin=[0,0,0]):
    binary_image = vtk.vtkImageData()

    binary_image.SetSpacing(spacing)
    binary_image.SetDimensions(dimensions)
    binary_image.SetExtent(0, dimensions[0]-1, 0, dimensions[1]-1, 0, dimensions[2]-1)

    binary_image.SetOrigin(origin)

    binary_image.AllocateScalars(vtk.VTK_FLOAT,1)

    N = binary_image.GetNumberOfPoints()
    for i in range(N):
        binary_image.GetPointData().GetScalars().SetTuple1(i,1.0)

    pol2stenc = vtk.vtkPolyDataToImageStencil()
    pol2stenc.SetInputData(polydata)
    pol2stenc.SetOutputOrigin(origin)
    pol2stenc.SetOutputSpacing(spacing)
    pol2stenc.SetOutputWholeExtent(binary_image.GetExtent())
    pol2stenc.Update()

    imgstenc = vtk.vtkImageStencil()
    imgstenc.SetInputData(binary_image)
    imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
    imgstenc.ReverseStencilOff()
    imgstenc.SetBackgroundValue(0.0)
    imgstenc.Update()

    return imgstenc.GetOutput()
