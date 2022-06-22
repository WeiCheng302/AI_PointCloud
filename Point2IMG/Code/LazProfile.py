import numpy as np
import math
import DEMFilter
import ARSEMimgTW as ARSEMimg
import ARSEMLazTW as ARSEMLaz
from numba import njit
from os.path import basename

lazname = 'E:\project_data/apply\Plane/94181061_reclass.las'
outfolder = 'D:\Point2IMG\Taiwan\profile_test/'

@njit(nogil= True)
def CheckBorder(value, masksize, thres):

    outValueMinus = value - masksize
    if outValueMinus < 0 : outValueMinus = 0

    outValuePosit = value + masksize
    if outValuePosit > thres : outValuePosit = thres

    return outValueMinus, outValuePosit

#@njit(nogil= True)
def RegionMin(imgH, imgW, img):

    ori = img.copy()

    for row in range(imgH):
        for col in range(imgW):

            ULrow, BRrow = CheckBorder(row, 10, imgH)
            ULcol, BRcol = CheckBorder(col, 10, imgW)
            min = np.min(ori[ULrow:BRrow, ULcol:BRcol])

            img[row, col] = ori[row, col] - min

    return img

@njit(nogil= True)
def AvgElevImg(ElevImg, imgsize, pixsize):

    # Input : Average Elevation imahe of every 1m * 1m image
    # Output : The averaged-elevation image for every profile image
    # Clip the original image into strips with the width at clipW
    # Caculate the average Elevation within every clipW

    clipW = math.ceil(imgsize * pixsize)
    meanH = ElevImg.mean()
    outImg = np.zeros((ElevImg.shape[0], ElevImg.shape[1]//clipW))

    for row in range(outImg.shape[0]):
        for col in range(outImg.shape[1]):

            sum = np.array([0])

            for num in ElevImg[row, (clipW * col) : (clipW * (col + 1))] :
                if num != 0 : np.append(sum, num)

            if len(sum) == 1 : outImg[row, col] = meanH
            else : outImg[row, col] = sum.sum() / (len(sum) - 1)

    return outImg

@njit(nogil= True)
def ProfileGeneration(Ep, Np, Hp, intensity, classification, Adj_elev_img_simp, imgH, imgsize, pixsize):

    ProfileStack = np.zeros((imgH, 1, 3, imgsize, imgsize), dtype=np.float32) # Elev, Int, Region Elev, Classification
    LabelStack = np.ones((imgH, 1, 1, imgsize, imgsize), dtype=np.uint8)

    for idx in range(len(Ep)):

        if int(Np[idx]) == imgH : continue
        avgElev = Adj_elev_img_simp[int(Np[idx]), int((Ep[idx]//imgsize))]

        profile_row = imgsize - int((Hp[idx] - avgElev)//pixsize)
        if (profile_row > 200) or (profile_row < -55) : continue

        profile_stack_col = Ep[idx]//imgsize
        if profile_stack_col != 0 : continue
        profile_col = int((Ep[idx]%imgsize)/pixsize)

        # if label == ground : continue
        #   if label == outlier : continue
        #   if ground : label ground(2)   else : label non-ground(0)

        if LabelStack[int(Np[idx]), 0, 0, profile_row, profile_col] != 2 :

            if classification[idx] == 30 : continue
            if classification[idx] == 2 : LabelStack[int(Np[idx]), 0, 0, profile_row, profile_col] = 2
            else : LabelStack[int(Np[idx]), 0, 0, profile_row, profile_col] = 0

            ProfileStack[int(Np[idx]), 0, 0, profile_row, profile_col] = Hp[idx]
            ProfileStack[int(Np[idx]), 0, 1, profile_row, profile_col] = intensity[idx]
            ProfileStack[int(Np[idx]), 0, 2, profile_row, profile_col] = Region_Min[int(Np[idx]), int(Ep[idx]//imgsize)]
        
    return ProfileStack, LabelStack

if __name__ == "__main__":

    imgsize = 256
    pixsize = 0.2

    laz = ARSEMLaz.ReadLAZ(lazname)
    Ep, Np, Hp, intensity, classification = ARSEMLaz.GetLAZInfo(laz)
    imgW, imgH, xymaxmin= ARSEMLaz.GetImgProjectionInfo(Ep, Np, Hp)
    Ep, Np = ARSEMLaz.PointCloudShift(Ep, Np, xymaxmin[3], xymaxmin[4])

    print("Caculating Required Information")
    GP_Container = DEMFilter.Reliable_Ground_points(Ep, Np, Hp, classification, imgW, imgH)        
    Adj_elev_img = DEMFilter.Elev_adjustment(imgW, imgH, GP_Container)
    Adj_elev_img_simp = AvgElevImg(Adj_elev_img[0], imgsize, pixsize) # Avg_Elev, for profile height
    Region_Min = RegionMin(imgH, imgW, Adj_elev_img[0]) # Local Minima

    print("Filling")
    ProfileStack, LabelStack = ProfileGeneration(Ep, Np, Hp, intensity, classification, Adj_elev_img_simp, imgH, imgsize, pixsize)

    print('Saving')
    for i in range(len(ProfileStack)):
        for j in range(len(ProfileStack[0])):

            filename = basename(lazname) + '_' + str(i) + '_' + str(j)

            ARSEMimg.SavePNG(outfolder + 'png/' + filename + '.png', LabelStack[i, j, 0])
            ARSEMimg.SaveTiff_1inp(outfolder + 'tif/' + filename + '.tif', ProfileStack[i, j], 0)

        