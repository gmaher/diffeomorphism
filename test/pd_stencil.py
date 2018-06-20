import vtk
import sys
import os
sys.path.append(os.path.abspath('..'))
from src.vtkUtil import polydata_to_binary_image, readVTKPD, read_mha, vtk_image_to_numpy

pd_string  = '/home/marsdenlab/datasets/vascular_data/OSMSC0110/0110_0001/0110_0001-cm.vtp'
img_string = '/home/marsdenlab/datasets/vascular_data/OSMSC0110/OSMSC0110-cm.mha'

mha = read_mha(img_string)
pd  = readVTKPD(pd_string)

binary_image = polydata_to_binary_image(pd, mha.GetDimensions(),
    mha.GetSpacing(), mha.GetOrigin())

writer = vtk.vtkMetaImageWriter()
writer.SetFileName("../data/binary_image.mha")
writer.SetInputData(binary_image)
writer.Write()

numpy_image = vtk_image_to_numpy(binary_image)
