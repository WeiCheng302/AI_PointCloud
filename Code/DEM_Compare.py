from math import sqrt
import ARSEMimgTW as ARSEMimg
import numpy as np
from osgeo import gdal

def waterInfo(name):

    ds = gdal.Open(name)
    gt = ds.GetGeoTransform()

    Xsize = ds.RasterXSize
    Ysize = ds.RasterYSize
    x_offset = gt[0]
    y_offset = gt[3]

    return np.array([Ysize, Xsize, x_offset, y_offset])

inp_dem = 'D:\Point2IMG\Taiwan/0725_All_2m/95183019_DiffImh.tif'
# ref_dem = 'D:\Point2IMG\Taiwan\Taiwan_Point_Cloud/103_plane\dem/DEMe94181061-0916_NNp2.tif'
# water = 'D:\Point2IMG\Taiwan\Taiwan_Point_Cloud/103_plane\waterlineTIF_105_110/94181061w.tif'

# wInfo = waterInfo(water)
# demInfo = waterInfo(inp_dem)

inp_dem = ARSEMimg.ReadImg(inp_dem)
# ref_dem = ARSEMimg.ReadImg(ref_dem)
# water = ARSEMimg.ReadImg(water)


# def WaterPadding(inp_dem, water, wInfo, demInfo):

#     water_mask = np.zeros(inp_dem.shape)
#     water_mask_out = np.zeros(inp_dem.shape) + 1

#     offx = int(wInfo[2] - demInfo[2])
#     #offy = int(demInfo[3] - wInfo[3])

#     water_mask[ 1 :  , offx + 1 : ] = water

#     water_mask[water_mask!=0] = 1

#     water_mask_out = water_mask_out - water_mask

#     return water_mask_out

# water_mask = WaterPadding(inp_dem, water, wInfo, demInfo)

# water_mask = np.ones(water_mask.shape)

# water_len = len(water_mask[water_mask!=0])
# diff = (inp_dem - ref_dem) * water_mask

# ARSEMimg.SaveTiff_1inp('D:\Point2IMG\Taiwan/0523_Plane\DEM/94181061_diff_GT_NW_water01.tif', diff, 0)
diff = inp_dem
print("Max : " + str(diff.max()))
print("Min : " + str(diff.min()))
print("Mean : " + str(diff.mean()))
print("Median : " + str(np.median(diff)))
print("Std : " + str(diff.std()))
print("RMSE : " + str(sqrt( (diff * diff).sum() / (diff.shape[0]*diff.shape[1]) )))
#print(sqrt(diff * diff).sum()/(water_len))

# print('=============')

# a = len(diff[diff>5])
# diff[diff>5] = 0
# a = a + len(diff[diff<-5])
# diff[diff<-5] = 0

# #ARSEMimg.SaveTiff_1inp('D:\Point2IMG\Taiwan/0523_Plane\DEM/94181061_diff_GT_water5_M.tif', diff, 0)
# print('=============')

# print(diff.max())
# print(diff.min())
# print(diff[diff!=0].mean())
# print(np.median(diff[diff!=0]))
# print(diff[diff!=0].std())
# print(sqrt( (diff * diff).sum() / (diff.shape[0]*diff.shape[1] - a) ) )
# #print(sqrt(diff * diff).sum()/(water_len))

# print('=============')
# a = a + len(diff[diff>1])
# diff[diff>1] = 0
# a = a + len(diff[diff>1])
# diff[diff<-1] = 0
# a = a + len(diff[diff>1])
# #ARSEMimg.SaveTiff_1inp('D:\Point2IMG\Taiwan/0523_Plane\DEM/94181061_diff_GT_water2_M.tif', diff, 0)
# print('=============')
# print(diff.max())
# print(diff.min())
# print(diff[diff!=0].mean())
# print(np.median(diff[diff!=0]))
# print(diff[diff!=0].std())
# print(sqrt( (diff * diff).sum() / (diff.shape[0]*diff.shape[1] - a) ))
# print(1 - (a/(diff.shape[0]*diff.shape[1])))
#print(sqrt(diff * diff).sum()/(water_len))
#print((diff * diff).sum()/(diff.shape[0]*diff.shape[1] - water_len))

