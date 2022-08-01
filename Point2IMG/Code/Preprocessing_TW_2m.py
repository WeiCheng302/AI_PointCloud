from multiprocessing.connection import wait
import os
import time
import shutil
import glob
import ARSEMLazTW as ARSEMLaz
import ARSEMimgTW as ARSEMimg
import numpy as np
from numba import njit, prange
from os.path import basename
from osgeo import gdal
from DEMFilter import LasNoiseFilter

LAZhome = "E:\project_data/apply\GroundObject/LowVeg"  #City #LowVeg #Forest #Testing
mother_folder = "D:\Point2IMG\Taiwan/0725_2m"

water_folder = "D:\Point2IMG\Taiwan\Taiwan_Point_Cloud/103_plane\waterlineTIF_105_110/"

ClipH = 256
ClipW = 256

fillin = 9999

Folder_UnClipped = mother_folder + "/UnClipped_Img/"
Folder_UnClippedPng = mother_folder + "/UnClipped_Png/"

Folder_UnClippedLabel_All = mother_folder + "/UnClipped_Png_All/"
Folder_UnClippedtrain_All = mother_folder + "/UnClipped_Img_All/"

FolderTif_Train = mother_folder + "/Train_Img/"
FolderPng_Train = mother_folder + "/Train_Label/"

FolderTif_Test = mother_folder + "/Test_Img/"
FolderPng_Test = mother_folder + "/Test_Label/"

FolderTif_Val = mother_folder + "/Val_Img/"
FolderPng_Val = mother_folder + "/Val_Label/"

FolderTif_All = mother_folder + "/All_Img/"
FolderPng_All = mother_folder + "/All_Label/"

FolderTif_Border = mother_folder + "/Border_Img/"
FolderPng_Border = mother_folder + "/Border_Label/"

FolderFilteredPointCloud = mother_folder + "/Filtered_PointCloud/"

@njit(nogil= True)
def CheckBorder(value, masksize, thres):

    outValueMinus = value - masksize
    if outValueMinus < 0 : outValueMinus = 0

    outValuePosit = value + masksize
    if outValuePosit > thres : outValuePosit = thres

    return outValueMinus, outValuePosit

@njit(nogil= True)
def RegionMin_and_WaterCover(imgH, imgW, img, classimg, xymaxmin, waterImg, wInfoList, fillin, water=False):

    ori = img.copy()
    imgH = int(imgH/2)
    imgW = int(imgW/2)

    for row in range(imgH):
        for col in range(imgW):

            # 3rd Band
            ULrow, BRrow = CheckBorder(row, 5, imgH)
            ULcol, BRcol = CheckBorder(col, 5, imgW)
            min = np.min(ori[0, ULrow:BRrow, ULcol:BRcol])

            if ori[0, row, col] != fillin : img[2, row, col] = ori[0, row, col] - min
 
            offx = int((wInfoList[2] - xymaxmin[3])/2)
            offy = int((xymaxmin[1] - wInfoList[3])/2)
                
            if 0 < (row - offy) < int(wInfoList[0]/2) :
                if 0 < (col - offx) < int(wInfoList[1]/2) :
                    if waterImg[2*(row - offy), 2*(col - offx)] == 1:
                        classimg[0, row, col] = 1

    return img, classimg

def waterInfo(name):

    ds = gdal.Open(name)
    gt = ds.GetGeoTransform()

    Xsize = ds.RasterXSize
    Ysize = ds.RasterYSize
    x_offset = gt[0]
    y_offset = gt[3]

    return np.array([Ysize, Xsize, x_offset, y_offset])

def LAZtoIMG(LAZname, TrainImgName, LabelImgName, NoiseFilt = False):

    #Image Generation
    print("Reading Point Cloud")
    laz = ARSEMLaz.ReadLAZ(LAZname)

    if NoiseFilt == True :
        start = time.time() 
        laz = LasNoiseFilter(laz, FolderFilteredPointCloud + basename(LAZname)[0:-4] + "_reclass.las")
        print(time.time() - start)

    print("Point Cloud Filtered")
    xp, yp, zp, intensity, classification = ARSEMLaz.GetLAZInfo(laz)
    imgW, imgH , xymaxmin= ARSEMLaz.GetImgProjectionInfo(xp, yp, zp)

    xp, yp = ARSEMLaz.PointCloudShift(xp, yp, xymaxmin[3], xymaxmin[4])

    try :
        waterImg = ARSEMimg.ReadImg(water_folder + basename(LAZname)[0:-12] + "w.tif", nodata0 = True)
        waterInfoList = waterInfo(water_folder + basename(LAZname)[0:-12] + "w.tif")#, xymaxmin[3], xymaxmin[4])
        water = True
    except AttributeError :
        waterImg = np.zeros((10, 10))
        waterInfoList = np.zeros((4))
        water = False

    print("    Setting Image")
    img = np.zeros((3, int(imgH/2), int(imgW/2)), dtype=np.float32) + fillin
    classimg = np.ones((1, int(imgH/2), int(imgW/2)), dtype=np.uint8)

    print("To 0")
    img, classimg = ARSEMimg.PointCloudProjection_2m(img, classimg, xp, yp, zp, intensity, classification)

    print("    Filling")
    img, classimg = RegionMin_and_WaterCover(imgH, imgW, img, classimg, np.array(xymaxmin), waterImg, waterInfoList, fillin, water)
    
    print("    Interpolating")
    img = ARSEMimg.AdaptiveBilinearInterpolation(img, classimg[0])

    print("    Saving")
    ARSEMimg.SaveTiff_1inp_Coord(TrainImgName, img, 0, xymaxmin)
    print("    TIFF Sucessfully Saved")

    #ARSEMimg.SaveTiff_1inp(LabelImgName, classimg, 0)
    ARSEMimg.SavePNG(LabelImgName, classimg[0])
    print("    PNG Sucessfully Saved")

    print("Number of 0 : " + str(len(classimg[classimg == 0])))
    print("Number of 1 : " + str(len(classimg[classimg == 1])))
    print("Number of 2 : " + str(len(classimg[classimg == 2])))

    return img, classimg

def ImageClip(img, classimg, LAZname):

    if type(img) == type("String"):
        Png = ARSEMimg.ReadImg(classimg)
        Tiff = ARSEMimg.ReadImg(img)
    else :
        Png = classimg[0]
        Tiff = img

    bands = Tiff.shape[0]

    print("Clipping")
    TiffList, TiffIDList, BTiffList, BTiffIDList = ARSEMimg.ClipTif_Border_Overlap(Tiff, ClipH, ClipW)
    PngList, PngIDList, BPngList, BPngIDList = ARSEMimg.ClipPng_Border_Overlap(Png, ClipH, ClipW)

    outImgName = basename(LAZname)[0:-4]

    print("Saving All Images")
    for i in range(len(PngList)):
        ARSEMimg.SaveTiff_ninp(FolderTif_All + outImgName + TiffIDList[bands * i] + ".tif", TiffList, i, bands, 0)
        ARSEMimg.SavePNG(FolderPng_All + outImgName + PngIDList[i] + ".png", PngList[i])

    for i in range(len(BPngIDList)):
        ARSEMimg.SaveTiff_ninp(FolderTif_All + outImgName + BTiffIDList[bands * i] + ".tif", BTiffList, i, bands, 0)
        ARSEMimg.SavePNG(FolderPng_All + outImgName + BPngIDList[i] + ".png", BPngList[i])
        
        ARSEMimg.SaveTiff_ninp(FolderTif_Border + outImgName + BTiffIDList[bands * i] + ".tif", BTiffList, i, bands, 0)
        ARSEMimg.SavePNG(FolderPng_Border + outImgName + BPngIDList[i] + ".png", BPngList[i])

if __name__ == "__main__":

    Lazfile = glob.glob(LAZhome + "/*.las")
    exist = glob.glob(Folder_UnClippedPng + "*.png")
    for i in range(len(exist)) : exist[i] = basename(exist[i])
    print(exist)
    exclude = ['94191036']

    for LAZname in Lazfile:
        print("Now Processing : " + LAZname)
        
        #if basename(LAZname)[:8] not in ['94181061', '95203004'] :continue
        if (basename(LAZname)[:8] in exclude) or ( (basename(LAZname)[:-4] + ".png") in exist):
            print("Skip " + LAZname)
            continue

        TrainImgName = Folder_UnClipped + basename(LAZname)[0:-4] + ".tif"
        LabelImgName = Folder_UnClippedPng + basename(LAZname)[0:-4] + ".png"

        TrainImgName_All = Folder_UnClippedtrain_All + basename(LAZname)[0:-4] + ".tif"
        LabelImgName_All = Folder_UnClippedLabel_All + basename(LAZname)[0:-4] + ".png"

        img, classimg = LAZtoIMG(LAZname, TrainImgName, LabelImgName, NoiseFilt = False)
        shutil.copyfile(TrainImgName, TrainImgName_All)
        shutil.copyfile(LabelImgName, LabelImgName_All)

        #ImageClip(img, classimg, LAZname)