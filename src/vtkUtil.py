import vtk

def mesh2polydata(mesh):
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

def saveMeshPolydata(mesh, filepath):
    pd = mesh2polydata(mesh)

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(filepath)
    writer.SetInputData(pd)
    writer.SetFileTypeToASCII()
    writer.Write()
