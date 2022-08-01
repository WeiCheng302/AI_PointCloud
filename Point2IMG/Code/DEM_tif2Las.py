import laspy
import ARSEMimgTW
import numpy as np
from osgeo import gdal
from os.path import basename

tif = "D:/Point2IMG/Taiwan/Test_DEM2Las/DEMe94182060-0930_NNp2.tif"
lasname = "D:/Point2IMG/Taiwan/Test_DEM2Las/" + basename(tif)[0:-4] + '.las'

GeoTransform = gdal.Open(tif).GetGeoTransform()
tif = ARSEMimgTW.ReadImg(tif)
size = tif.shape[0] * tif.shape[1]

x = np.zeros(size)
y = np.zeros(size)
z = np.zeros(size)

for i in range(tif.shape[0]):
    for j in range(tif.shape[1]):
        x[i * tif.shape[1] + j] = GeoTransform[0] + GeoTransform[1] * j
        y[i * tif.shape[1] + j] = GeoTransform[3] + GeoTransform[5] * i
        z[i * tif.shape[1] + j] = tif[i, j]

header = laspy.header.LasHeader()
laz_out = laspy.create(point_format=header.point_format, file_version=header.version)
laz_out.x = x 
laz_out.y = y
laz_out.z = z
laz_out.write(lasname)