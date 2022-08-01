import laspy
import ARSEMLazTW
import ARSEMimgTW
import numpy as np
from osgeo import gdal
from os.path import basename

demtif = "D:\Point2IMG\Taiwan\Test_DEM2Las/DEMe94181065-0930_NNp2.tif"
in_las = "D:\Point2IMG\Taiwan\Test_DEM2Las/94181065.las"
out_lasname = "D:/Point2IMG/Taiwan/Test_DEM2Las/94181065_DEMMerge.las"

#Get DEMTiff GeoTransform
GeoTransform = gdal.Open(demtif).GetGeoTransform()
demtif = ARSEMimgTW.ReadImg(demtif)
size = demtif.shape[0] * demtif.shape[1]

#Read Original LAS
in_las = ARSEMLazTW.ReadLAZ(in_las)
xplas = np.array(in_las.x)
yplas = np.array(in_las.y)
zplas = np.array(in_las.z)
cllas = np.array(in_las.classification)

#DEM as Point Cloud
xpdem = np.zeros(size)
ypdem = np.zeros(size)
zpdem = np.zeros(size)

for i in range(demtif.shape[0]):
    for j in range(demtif.shape[1]):
        xpdem[i * demtif.shape[1] + j] = GeoTransform[0] + GeoTransform[1] * j
        ypdem[i * demtif.shape[1] + j] = GeoTransform[3] + GeoTransform[5] * i
        zpdem[i * demtif.shape[1] + j] = demtif[i, j]

#Merging the DEM data and Point Cloud Data
length = len(xplas) + len(xpdem)

outX = np.zeros(length)
outY = np.zeros(length)
outZ = np.zeros(length)
outCL = np.zeros(length) + 5

outX[0:len(xplas)] = xplas
outY[0:len(yplas)] = yplas
outZ[0:len(zplas)] = zplas
outCL[0:len(cllas)] = cllas

outX[len(xplas) : ] = xpdem
outY[len(yplas) : ] = ypdem
outZ[len(zplas) : ] = zpdem

#Data Output
header = laspy.header.LasHeader()
laz_out = laspy.create(point_format=header.point_format, file_version=header.version)
laz_out.x = outX 
laz_out.y = outY
laz_out.z = outZ
laz_out.classification = outCL
laz_out.write(out_lasname)