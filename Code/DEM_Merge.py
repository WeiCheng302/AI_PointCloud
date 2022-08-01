import ARSEMLazTW
import laspy
import numpy as np
las = "D:/Point2IMG/Taiwan/Test_DEM2Las/94182060_Clip.las"
dem = "D:/Point2IMG/Taiwan/Test_DEM2Las/DEMe94182060-0930_NNp2_Clip2.las"

las = ARSEMLazTW.ReadLAZ(las)
dem = ARSEMLazTW.ReadLAZ(dem)

xplas = np.array(las.x)
yplas = np.array(las.y)
zplas = np.array(las.z)
cllas = np.array(las.classification)

xpdem = np.array(dem.x)
ypdem = np.array(dem.y)
zpdem = np.array(dem.z)

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

header = laspy.header.LasHeader()
laz_out = laspy.create(point_format=header.point_format, file_version=header.version)
laz_out.x = outX 
laz_out.y = outY
laz_out.z = outZ
laz_out.classification = outCL
laz_out.write("D:/Point2IMG/Taiwan/Test_DEM2Las/94182060_DEMfilt_Merge3.las")