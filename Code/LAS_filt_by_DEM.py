import glob
import numpy as np
import ARSEMimgTW as ARSEMimg
import ARSEMLazTW as ARSEMLaz
from os.path import basename
from numba import njit
from osgeo import gdal

@njit(nogil=True)
def LasDemFilter(dem, xp, yp, zp, classification, prior = 0.048):

    count = 0
    imgH = dem.shape[0] - 1
    imgW = dem.shape[1] - 1
    
    for i in range(len(xp)):
 
        if (imgH - int(yp[i])) < imgH and int(xp[i]) < imgW :
            ElevCorner = (dem[imgH - int(yp[i]), int(xp[i])] + dem[imgH - int(yp[i]) + 1, int(xp[i])] 
                        + dem[imgH - int(yp[i]), int(xp[i]) + 1] + dem[imgH - int(yp[i]) + 1, int(xp[i]) + 1]) * 0.25
        else : ElevCorner = dem[imgH - int(yp[i]), int(xp[i])]
        # print(ElevCorner)
        # print(zp[i])
        # print("=======")
        if classification[i] == 2 : continue
        if ElevCorner - 2 * prior < zp[i] < ElevCorner + 2 * prior : 
            classification[i] = 2
            count += 1
        else : classification[i] = 31

    return classification, count

def LAS_DEM_Preprocess(lasname, demname, outlasname):

    dem = ARSEMimg.ReadImg(demname)
    laz = ARSEMLaz.ReadLAZ(lasname)
    tiffInfo = ARSEMimg.TIFFInfo(demname)

    xp, yp, zp, intensity, classification = ARSEMLaz.GetLAZInfo(laz)
    imgW, imgH , xymaxmin= ARSEMLaz.GetImgProjectionInfo(xp, yp, zp)
    xp, yp = ARSEMLaz.PointCloudShift(xp, yp, xymaxmin[3], xymaxmin[4])
    #intensity = ARSEMLaz.Normalization(intensity, intensity.std(), 0)
    classification, count = LasDemFilter(np.array(dem), xp, yp, zp, classification, prior = 0.048)

    print(count)
    print('Writing')
    laz = ARSEMLaz.WriteLAZ_int(laz, xp+xymaxmin[3], yp+xymaxmin[4], zp, intensity, classification, outlasname)

    return laz

if __name__ == "__main__":

    FolderDEM = 'D:\Point2IMG\Taiwan\Taiwan_Point_Cloud/103_plane\dem/'
    inLasFolder = 'E:\project_data/apply\Plane/Raw/'
    outLasFolder = 'E:\project_data/apply\Plane\DEMfilt/'

    DEMList = glob.glob(FolderDEM + '*.tif')
    lasnameList = glob.glob(inLasFolder + '*.las')
    #lasnameList = ['E:\project_data/apply\Mont\Raw/95191003.las']
    exist = ['94191036.las'] #glob.glob(outLasFolder + '*.las')

    for lasname in lasnameList:

        print('Now Processing : ' + lasname)
        # print(outLasFolder + basename(lasname)[:8] + '_DEMfilt.las')
        if (basename(lasname)[:8] + '.las') in exist : continue

        for filename in DEMList:

            if basename(filename)[4:12] == basename(lasname)[:8]:
                DemName = filename
                print(DemName)
                break

        laz = LAS_DEM_Preprocess(lasname, DemName, outLasFolder + basename(lasname)[:8] + '_DEMfilt.las')
        






'''
FolderDEM = 'D:\Point2IMG\Taiwan\Taiwan_Point_Cloud/103_plane\dem/'
DEMList = glob.glob(FolderDEM + '*.tif')

lasname = 'E:\project_data/apply\Hill\Raw/94192045.las'
#lasname = 'E:\project_data/apply\Mont\Raw/95191003.las'
#lasname = 'E:\project_data/apply\City\Raw/95213071.las'
#lasname = 'E:\project_data/apply\Mont\Raw/95183040.las'
#lasname = 'E:\project_data/apply\Plane\Raw/94182060.las'

for filename in DEMList:
    if basename(filename)[4:12] == basename(lasname)[:8]:
        DemName = filename
        break

dem = ARSEMimg.ReadImg(DemName)
tiffInfo = ARSEMimg.TIFFInfo(DemName)
print(dem)
print(tiffInfo)
laz = ARSEMLaz.ReadLAZ(lasname)

xp, yp, zp, intensity, classification = ARSEMLaz.GetLAZInfo(laz)
imgW, imgH , xymaxmin= ARSEMLaz.GetImgProjectionInfo(xp, yp, zp)
xp, yp = ARSEMLaz.PointCloudShift(xp, yp, xymaxmin[3], xymaxmin[4])
intensity = ARSEMLaz.Normalization(intensity, intensity.std(), 0)

fillin = 9999
img = ARSEMimg.CreateImgArray(imgH, imgW, 3, np.float32) + fillin
classimg = ARSEMimg.CreateImgArray(imgH, imgW, 1, np.uint8) + fillin
a = ARSEMimg.PointCloudProjection(img, classimg, xp, yp, zp, intensity, classification)
print('Filtering')
classification, count = LasDemFilter(np.array(dem), xp, yp, zp, classification, prior = 0.048)
print(count)
print('Writing')
ARSEMLaz.WriteLAZ_int(laz, xp+xymaxmin[3], yp+xymaxmin[4], zp, intensity, classification, '95191003_DEMfilt.las')
'''