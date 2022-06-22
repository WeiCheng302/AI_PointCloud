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
#from AfterProcessing import MergeImg

LAZhome = "E:\project_data/apply\Plane/Raw" #"E:\project_data/apply\City\DEMfilt/Pure"
mother_folder = "D:\Point2IMG\Taiwan/0615_Interpolation"

water_folder = "D:\Point2IMG\Taiwan\Taiwan_Point_Cloud/103_plane\waterlineTIF_105_110/"

ClipH = 256
ClipW = 256

fillin = 9999

Folder_UnClipped = mother_folder + "/UnClipped_Img/"
Folder_UnClippedPng = mother_folder + "/UnClipped_Png/"

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

    for row in range(imgH):
        for col in range(imgW):

            # 3rd Band
            ULrow, BRrow = CheckBorder(row, 10, imgH)
            ULcol, BRcol = CheckBorder(col, 10, imgW)
            min = np.min(ori[0, ULrow:BRrow, ULcol:BRcol])

            if ori[0, row, col] != fillin : img[2, row, col] = ori[0, row, col] - min
            
            # 4th Band
            # ULrow, BRrow = CheckBorder(row, 3, imgH)
            # ULcol, BRcol = CheckBorder(col, 3, imgW)
            # min = np.min(ori[0, ULrow:BRrow, ULcol:BRcol])

            # if ori[0, row, col] != fillin : img[3, row, col] = ori[0, row, col] - min

            # 5th Band
            # ULrow, BRrow = CheckBorder(row, 50, imgH)
            # ULcol, BRcol = CheckBorder(col, 50, imgW)
            # min = np.min(ori[0, ULrow:BRrow, ULcol:BRcol])

            # if ori[0, row, col] != fillin : img[4, row, col] = ori[0, row, col] - min

            # Unused Band
            #if img[0, row, col] != fillin : img[3, row, col] = 1
            #else : img[3, row, col] = 0

            #if img[0, row, col] == fillin : img[:, row, col] = -0.1   
            #if classimg[0, row, col] == fillin : classimg[0, row, col] = 1

            #Water Check
            #if water: 
            offx = int(wInfoList[2] - xymaxmin[3])
            offy = int(xymaxmin[1] - wInfoList[3])
                
            if 0 < (row - offy) < wInfoList[0] :
                if 0 < (col - offx) < wInfoList[1] :
                    if waterImg[(row - offy), (col - offx)] == 1:
                        #img[:, row, col] = -0.1
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
    
    #wXsize, wYSize, wx_offset, wy_offset 

    if NoiseFilt == True :
        start = time.time() 
        laz = LasNoiseFilter(laz, FolderFilteredPointCloud + basename(LAZname)[0:-4] + "_reclass.las")
        print(time.time() - start)

    print("Point Cloud Filtered")
    xp, yp, zp, intensity, classification = ARSEMLaz.GetLAZInfo(laz)
    imgW, imgH , xymaxmin= ARSEMLaz.GetImgProjectionInfo(xp, yp, zp)

    xp, yp = ARSEMLaz.PointCloudShift(xp, yp, xymaxmin[3], xymaxmin[4])
    #zp = ARSEMLaz.Normalization(zp, xymaxmin[2], xymaxmin[5])
    #intensity = ARSEMLaz.Normalization(intensity, intensity.std(), 0)
    #intensity = ARSEMLaz.Normalization(intensity, 65536, 0)    
    try :
        waterImg = ARSEMimg.ReadImg(water_folder + basename(LAZname)[0:-12] + "w.tif", nodata0 = True)
        waterInfoList = waterInfo(water_folder + basename(LAZname)[0:-12] + "w.tif")#, xymaxmin[3], xymaxmin[4])
        water = True
    except AttributeError :
        waterImg = np.zeros((10, 10))
        waterInfoList = np.zeros((4))
        water = False

    print("    Setting Image")
    # img = ARSEMimg.CreateImgArray(imgH, imgW, 3, np.float32) + fillin
    # classimg = ARSEMimg.CreateImgArray(imgH, imgW, 1, np.uint8) + fillin
    img = np.zeros((3, imgH, imgW), dtype=np.float32) + fillin
    classimg = np.ones((1, imgH, imgW), dtype=np.uint8)

    print("To 0")
    img, classimg = ARSEMimg.PointCloudProjection(img, classimg, xp, yp, zp, intensity, classification)

    print("    Filling")
    img, classimg = RegionMin_and_WaterCover(imgH, imgW, img, classimg, np.array(xymaxmin), waterImg, waterInfoList, fillin, water)
    
    print("    Interpolating")
    img = ARSEMimg.AdaptiveBilinearInterpolation(img, classimg[0])
    # mean0 = (img[0, :, :][img[0, :, :] != -0.1]).mean()
    # mean1 = (img[1, :, :][img[1, :, :] != -0.1]).mean()
    # mean2 = (img[2, :, :][img[2, :, :] != -0.1]).mean()

    # img[0, :, :][img[0, :, :] == -0.1] = mean0
    # img[1, :, :][img[1, :, :] == -0.1] = mean1
    # img[2, :, :][img[2, :, :] == -0.1] = mean2

    # img[0, :, :] = img[0, :, :] / (img[0][img[0] != mean0]).std()
    # img[1, :, :] = img[1, :, :] / (img[1][img[1] != mean1]).std()
    # img[2, :, :] = img[2, :, :] / (img[2][img[2] != mean2]).std()

    #img, classimg = ARSEMimg.Resample(img, classimg)

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
    TiffList, TiffIDList, BTiffList, BTiffIDList = ARSEMimg.ClipTif_Border(Tiff, ClipH, ClipW)
    PngList, PngIDList, BPngList, BPngIDList = ARSEMimg.ClipPng_Border(Png, ClipH, ClipW)

    outImgName = basename(LAZname)[0:-4]

    print("Saving All Images")
    for i in range(len(PngList)):
        ARSEMimg.SaveTiff_ninp(FolderTif_All + outImgName + TiffIDList[bands * i] + ".tif", TiffList, i, bands, 0)
        ARSEMimg.SavePNG(FolderPng_All + outImgName + PngIDList[i] + ".png", PngList[i])
        #ARSEMimg.SaveTiff_ninp(FolderPng_All + outImgName + PngIDList[i] + ".png", PngList, i, 1, 0)

    for i in range(len(BPngIDList)):
        ARSEMimg.SaveTiff_ninp(FolderTif_All + outImgName + BTiffIDList[bands * i] + ".tif", BTiffList, i, bands, 0)
        ARSEMimg.SavePNG(FolderPng_All + outImgName + BPngIDList[i] + ".png", BPngList[i])
        #ARSEMimg.SaveTiff_ninp(FolderPng_All + outImgName + BPngIDList[i] + ".png", BPngList, i, 1, 0)
        
        ARSEMimg.SaveTiff_ninp(FolderTif_Border + outImgName + BTiffIDList[bands * i] + ".tif", BTiffList, i, bands, 0)
        ARSEMimg.SavePNG(FolderPng_Border + outImgName + BPngIDList[i] + ".png", BPngList[i])
        #ARSEMimg.SaveTiff_ninp(FolderPng_Border + outImgName + BPngIDList[i] + ".png", BPngList, i, 1, 0)

def Seperate_Data(train_ratio = 0.8, val_ratio = 0.2, test_ratio = 0):

    assert train_ratio + val_ratio + test_ratio == 1, "Sum of the ratios should be 1."

    #FileInFolderTif = np.random.permutation(glob.glob(FolderTif_All + "*.tif"))
    FileInFolderLabel = np.random.permutation(glob.glob(FolderPng_All + "*.png"))

    print("Copying")
    lenList = (len(FileInFolderLabel))

    #Training
    print("    Saving Training Images")
    for i in range(int(lenList * train_ratio)):
        pngname = basename(FileInFolderLabel[i])
        shutil.copyfile(FolderTif_All + pngname[0:-4] + ".tif" ,FolderTif_Train + pngname[0:-4] + ".tif")
        shutil.copyfile(FileInFolderLabel[i] ,FolderPng_Train + pngname)
        
    #Validation
    print("    Saving Validation Images")
    for i in range(int(lenList * train_ratio), int(lenList * (train_ratio + val_ratio))):
        pngname = basename(FileInFolderLabel[i])
        shutil.copyfile(FolderTif_All + pngname[0:-4] + ".tif" ,FolderTif_Val + pngname[0:-4] + ".tif")
        shutil.copyfile(FileInFolderLabel[i] ,FolderPng_Val + pngname)

    #Testing
    print("    Saving Testing Images")
    for i in range(int(lenList * (train_ratio + val_ratio)), lenList):
        pngname = basename(FileInFolderLabel[i])
        shutil.copyfile(FolderTif_All + pngname[0:-4] + ".tif" ,FolderTif_Test + pngname[0:-4] + ".tif")
        shutil.copyfile(FileInFolderLabel[i] ,FolderPng_Test + pngname)        

if __name__ == "__main__":

    Lazfile = glob.glob(LAZhome + "/*.las")
    exist = glob.glob(Folder_UnClippedPng + "*.png")
    for i in range(len(exist)) : exist[i] = basename(exist[i])
    print(exist)
    exclude = ['94191036']

    for LAZname in Lazfile:
        print("Now Processing : " + LAZname)
        if basename(LAZname)[:8] not in ['94181061', '95203004'] :continue
        if (basename(LAZname)[:8] in exclude) or ( (basename(LAZname)[:-4] + ".png") in exist):
            print("Skip " + LAZname)
            continue
        TrainImgName = Folder_UnClipped + basename(LAZname)[0:-4] + ".tif"
        LabelImgName = Folder_UnClippedPng + basename(LAZname)[0:-4] + ".png"
        img, classimg = LAZtoIMG(LAZname, TrainImgName, LabelImgName, NoiseFilt = False)

        #classimg = classimg.reshape(1, classimg.shape[0], classimg.shape[1])

        ImageClip(img, classimg, LAZname)

    #Seperate_Data(train_ratio = 0.8, val_ratio = 0.2, test_ratio = 0)
    #MergeImg(LAZname, FolderPng_All, mother_folder)