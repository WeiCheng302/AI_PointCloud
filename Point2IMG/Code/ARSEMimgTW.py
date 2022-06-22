import math
from re import L, T
from math import ceil
import imageio
import numpy as np
from osgeo import gdal
from numba import njit, prange

#Image I/O

##Read Image
def ReadImg(path, nodata0 = False):

    data = gdal.Open(path)
    if nodata0:
        data.GetRasterBand(1).SetNoDataValue(0)
    img = data.ReadAsArray()
    return img

def TIFFInfo(name):

    ds = gdal.Open(name)
    gt = ds.GetGeoTransform()

    Xsize = ds.RasterXSize
    Ysize = ds.RasterYSize
    x_offset = gt[0]
    y_offset = gt[3]

    return np.array([Ysize, Xsize, x_offset, y_offset])

##Save Tiff, 1 input and Abs Coordinate
def SaveTiff_1inp_Coord(path, array, nodata, xymaxmin, water = False):

    nbands, nrows, ncols  = array.shape

    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(path, ncols, nrows, nbands, gdal.GDT_Float32)
    ds.SetGeoTransform( (int(xymaxmin[3]), 1, 0, int(xymaxmin[4] + nrows), 0, -1) )
    #ds.SetProjection("EPSG:3826")

    if nbands == 1 :
        band = ds.GetRasterBand(1)
        band.SetNoDataValue(nodata)
        band.WriteArray(array)
    else :
        for i in range(nbands):
            band = ds.GetRasterBand(i+1)
            band.SetNoDataValue(nodata)
            band.WriteArray(array[i])

##Save Tiff, 1 input
def SaveTiff_1inp(path, array, nodata):

    try:
        nbands, nrows, ncols  = array.shape

        driver = gdal.GetDriverByName("GTiff")
        ds = driver.Create(path, ncols, nrows, nbands, gdal.GDT_Float32)

        for i in range(nbands):
            band = ds.GetRasterBand(i+1)
            band.SetNoDataValue(nodata)
            band.WriteArray(array[i])

    except ValueError:
        nbands = 1
        nrows, ncols = array.shape

        driver = gdal.GetDriverByName("GTiff")
        ds = driver.Create(path, ncols, nrows, nbands, gdal.GDT_Float32)
    
        for i in range(nbands):
            band = ds.GetRasterBand(i+1)
            band.SetNoDataValue(nodata)
            band.WriteArray(array)

##Save Tiff, 3 input
def SaveTiff_3inp(path, array0, array1, array2, bands, nodata):
    nrows, ncols  = array1.shape
    array = [array0 * 1000, array1, array2]
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(path, ncols, nrows, bands, gdal.GDT_Int16)
    
    for i in range(bands):
        band = ds.GetRasterBand(i+1)
        band.SetNoDataValue(nodata)
        band.WriteArray(array[i])

def SaveTiff_4inp(path, array0, array1, array2, array3, bands, nodata):
    nrows, ncols  = array1.shape
    array = [array0, array1, array2, array3]
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(path, ncols, nrows, bands, gdal.GDT_Float32)
    
    for i in range(bands):
        band = ds.GetRasterBand(i+1)
        band.SetNoDataValue(nodata)
        band.WriteArray(array[i])

def SaveTiff_ninp(path, list, i, bands, nodata):
    nrows, ncols  = list[0].shape
    arrays = []

    if bands == 1:
        arrays.append(list[bands * i])
    else:
        for band in range(bands):
            arrays.append(list[bands * i + band])

    ds = gdal.GetDriverByName("GTiff").Create(path, ncols, nrows, bands, gdal.GDT_Float32)
    
    for i in range(bands):
        band = ds.GetRasterBand(i + 1)
        band.SetNoDataValue(nodata)
        band.WriteArray(arrays[i])

##Save Png
def SavePNG(path, array):
    imageio.imwrite(path, array)

#Image Initialization
def CreateImgArray(imgH, imgW, bands, datatype):
    Zeros = np.zeros(shape=(bands, imgH, imgW))
    Zeros = Zeros.astype(datatype)
    return Zeros

#Image Clipping

##Clip the image but give up the border
def ClipPng(Img, clipH, clipW):

    imgH, imgW = Img.shape

    subImgList = []
    subImgidList = []

    h = int(imgH/clipH)
    w = int(imgW/clipW)
    
    for i in range(h):
        for j in range(w):
            subImg = Img [ clipH * i : clipH * (i + 1), clipW * j : clipW * (j + 1)]
            subImgList.append(subImg)
            subImgidList.append("_" + str(i) + "_" + str(j))

    return subImgList, subImgidList


def ClipTif(Img, clipH, clipW):
    
    nbands, imgH, imgW = Img.shape

    subImgList = []
    subImgidList = []

    for i in range(int(imgH/clipH)):
        for j in range(int(imgW/clipW)):
            subImgidList.append("_" + str(i) + "_" + str(j))
            for k in range(nbands):
                subImg = Img[k, clipH * i : clipH * (i + 1), clipW * j : clipW * (j + 1)]
                subImgList.append(subImg)

    return subImgList, subImgidList

##Clip the image and keep the border
def ClipPng_Border(Img, clipH, clipW):

    imgH, imgW = Img.shape

    subImgList = []
    subImgidList = []
    BorderImgList = []
    BorderImgidList = []

    h = int(imgH/clipH)
    w = int(imgW/clipW)
    
    for i in range(h + 1):
        for j in range(w + 1):
            if i != h and j != w :
                subImg = Img [ clipH * i : clipH * (i + 1), clipW * j : clipW * (j + 1)]
                subImgList.append(subImg)
                subImgidList.append("_" + str(i) + "_" + str(j))
            elif i == h and j != w:
                borderImg = Img [ imgH - clipH : imgH, clipW * j : clipW * (j + 1)]
                BorderImgidList.append("_" + str(i) + "_" + str(j))
                BorderImgList.append(borderImg)
            elif i != h and j == w:
                borderImg = Img [ clipH * i : clipH * (i + 1), imgW - clipW : imgW]
                BorderImgidList.append("_" + str(i) + "_" + str(j))
                BorderImgList.append(borderImg)
            elif i == h and j == w: 
                borderImg = Img[imgH - clipH : imgH, imgW - clipW : imgW]
                BorderImgidList.append("_" + str(i) + "_" + str(j))
                BorderImgList.append(borderImg)

    return subImgList, subImgidList, BorderImgList, BorderImgidList

def ClipTif_Border(Img, clipH, clipW):
    
    nbands, imgH, imgW = Img.shape

    subImgList = []
    subImgidList = []
    BorderImgList = []
    BorderImgidList = []

    h = int(imgH/clipH)
    w = int(imgW/clipW) 
    
    for i in range(h + 1):
        for j in range(w + 1):
            if i != h and j != w :
                for k in range(nbands):
                    subImg = Img[k, clipH * i : clipH * (i + 1), clipW * j : clipW * (j + 1)]
                    subImgList.append(subImg) 
                    subImgidList.append("_" + str(i) + "_" + str(j))
            elif i == h and j != w:
                for k in range(nbands):
                    borderImg = Img[k, imgH - clipH : imgH, clipW * j : clipW * (j + 1)]
                    BorderImgidList.append("_" + str(i) + "_" + str(j))
                    BorderImgList.append(borderImg)
            elif i != h and j == w:
                for k in range(nbands):
                    borderImg = Img[k, clipH * i : clipH * (i + 1), imgW - clipW : imgW]
                    BorderImgidList.append("_" + str(i) + "_" + str(j))
                    BorderImgList.append(borderImg)
            elif i == h and j == w: 
                for k in range(nbands):
                    borderImg = Img[k, imgH - clipH : imgH, imgW - clipW : imgW]
                    BorderImgidList.append("_" + str(i) + "_" + str(j))
                    BorderImgList.append(borderImg)

    return subImgList, subImgidList, BorderImgList, BorderImgidList

#Set Image pixel Value

##Set the image value but ignore the border
@njit(nogil= True) #py:23s nopy,parral:9s nopy:8s
def SetImgValue(img, outImg, imgH, imgW, h, w, bands):
    for row in range(imgH):
        for column in range(imgW):
            if bands == 1: 
                outImg[h * imgH + row, w * imgW + column] = img[row, column]
            else:
                for layer in range(bands):
                    outImg[layer, h * imgH + row, w * imgW + column] = img[layer, row, column]
    return outImg

##Set the image value of the border
@njit(nogil= True)
def SetImgValueBorder(img, outImg, imgH, imgW, h, w, bands):
    for row in range(imgH):
        for column in range(imgW):
            if bands == 1:
                if h == 48 and w != 39:
                    outImg[0, outImg.shape[1] - img.shape[0] + row, w * imgW + column] = img[row, column]
                elif h != 48 and w == 39:
                    outImg[0, h * imgH + row, outImg.shape[2] - img.shape[1] + column] = img[row, column]
                elif h == 48 and w == 39:
                    outImg[0, outImg.shape[1] - img.shape[0] + row , outImg.shape[2] - img.shape[1] + column] = img[row, column]
            else:
                for layers in range(bands):
                    if h == 48 and w != 39:
                        outImg[layers, outImg.shape[1] - img.shape[1] + row, w * imgW + column] = img[layers, row, column]
                    elif h != 48 and w == 39:
                        outImg[layers, h * imgH + row, outImg.shape[2] - img.shape[2] + column] = img[layers, row, column]
                    elif h == 48 and w == 39:
                        outImg[layers, outImg.shape[1] - img.shape[1] + row , outImg.shape[2] - img.shape[2] + column] = img[layers, row, column]
    return outImg

##Set the image value and the border
@njit(nogil= True)
def SetImgValueAll(img, outImg, imgH, imgW, h, w):
    hcount = int(outImg.shape[1] / img.shape[0])
    wcount = int(outImg.shape[2] / img.shape[1])
    for row in range(imgH):
        for column in range(imgW):
            if h == hcount and w != wcount:
                outImg[0, outImg.shape[1] - img.shape[0] + row, w * imgW + column] = img[row, column]
            elif h != hcount and w == wcount:
                outImg[0, h * imgH + row, outImg.shape[2] - img.shape[1] + column] = img[row, column]
            elif h == hcount and w == wcount:
                outImg[0, outImg.shape[1] - img.shape[0] + row , outImg.shape[2] - img.shape[1] + column] = img[row, column]
            elif h != hcount and w != wcount:
                outImg[0, h * imgH + row, w * imgW + column] = img[row, column]
    return outImg

##Projecting LiDAR point cloud to Image
@njit(nogil=True)
def PointCloudProjection(train, label, xArr, yArr, zArr, intensityArr, classArr, groundLabel = 2, nongroundLabel = 0):

    imgH = train.shape[1] - 1

    for i in range(len(xArr)):

        if yArr[i] == train.shape[1] or xArr[i] == train.shape[2] : continue
    
        pixclass = classArr[i]  

        if zArr[i] < train[0, imgH - int(yArr[i]), int(xArr[i])]:
            train[0, imgH - int(yArr[i]), int(xArr[i])] = zArr[i]
            train[1, imgH - int(yArr[i]), int(xArr[i])] = intensityArr[i]
            if pixclass == 0 or pixclass == 1 or pixclass == 2 : label[0, imgH - int(yArr[i]), int(xArr[i])] = groundLabel
            elif pixclass == 6 or pixclass == 9 or pixclass == 31 : label[0, imgH - int(yArr[i]), int(xArr[i])] = nongroundLabel
    print("Image Value Sucessfully Set")
    return train, label

##Projecting the lowest elevation to the image
@njit(nogil= True, parallel = False)
def CreateMinImgTW(xp, yp, zp, xMin, yMin, minImg):

    imgH = minImg.shape[1] - 1
    
    for i in range(len(xp)):
        a = xp[i]
        if zp[i] < minImg[0, imgH - int(yp[i] - yMin), int(xp[i] - xMin)] : 
            minImg[0, imgH - int(yp[i] - yMin), int(xp[i] - xMin)] = zp[i]

    return minImg

#Mask Processing
def CheckBorder(value, masksize, thres):

    outValueMinus = value - masksize
    if outValueMinus < 0 : outValueMinus = 0

    outValuePosit = value + masksize
    if outValuePosit > thres : outValuePosit = thres

    return outValueMinus, outValuePosit

def GetRegionMin(img, ULrow, BRrow, ULcol, BRcol):
    return np.min(img[0, ULrow:BRrow, ULcol:BRcol])

#@njit(nogil= True)
def Ground80(img, thres):
    #groundcount = 0
    #for row in prange(img.shape[0]):#for col in prange(img.shape[1]):#if img[row, col] == 0 : groundcount += 1

    ratio = len(img[img == 1])/(img.shape[0] * img.shape[1])

    if ratio >= thres : return True
    else : return False

#For Testing ...
def TestImgDiff(img1, img2):

    diff = []
    
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if img1[i, j] - img2[i, j] != 0 : diff.append(img1[i, j] - img2[i, j])

    return(diff)

@njit(nogil= True)
def Resample(featimg, labelimg):

    outLabelimg = np.ones((ceil(labelimg.shape[1]/2), ceil(labelimg.shape[2]/2)), np.uint8)
    outFeatimg = np.zeros((featimg.shape[0], ceil(featimg.shape[1]/2), ceil(featimg.shape[2]/2)), np.float32) - 1

    for row in range(ceil(labelimg.shape[1]/2)):
        for col in range(ceil(labelimg.shape[2]/2)):
            
            isground = False
            ground_coord = np.array([-1, -1])
            ul = np.array([-1, -1])
            mask = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
 
            if row == int(labelimg.shape[1]/2) and row % 2 == 1 and col == int(labelimg.shape[2]/2) and col % 2 == 1:
                mask = np.array([[-1, -1]])
            elif row == int(labelimg.shape[1]/2) : #and row % 2 == 1:
                mask = np.array([[-1, 0], [-1, 1]])
            elif col == int(labelimg.shape[2]/2) : #and col % 2 == 1:
                mask = np.array([[0, -1], [1, -1]])

            for idx in mask:
                
                if labelimg[0, 2 * row + idx[0], 2 * col + idx[1]] != 1 : #and ul[0] != -1:
                    ul = idx

                if labelimg[0, 2 * row + idx[0], 2 * col + idx[1]] == 2:
                    isground = True
                    ground_coord = idx
                    break

            if not isground : 
                outLabelimg[row, col] = labelimg[0, 2 * row + ul[0], 2 * col + ul[0]]
                outFeatimg[:, row, col] = featimg[:, 2 * row + ul[0], 2 * col + ul[0]]
            else : 
                outLabelimg[row, col] = labelimg[0, 2 * row + ground_coord[0], 2 * col + ground_coord[1]]
                outFeatimg[:, row, col] = featimg[:, 2 * row + ground_coord[0], 2 * col + ground_coord[1]]

    return outFeatimg, outLabelimg
            
@njit(nogil= True)
def AdaptiveBilinearInterpolation(featimg, classimg):
    out_featimg = featimg.copy()
    bands = featimg.shape[0]
    imgH = featimg.shape[1]
    imgW = featimg.shape[2]
    for band in range(bands):
        for row in range(imgH):
            for col in range(imgW):
                if classimg[row, col] == 1:
                    # Get the items to interpolate 
                    intp_row, intp_col, intp_pixvalue = GetBilinearElement(featimg[band], classimg, row, col)
                    out_featimg[band, row, col] = BilinearInterpolation(row, col, intp_row, intp_col, intp_pixvalue)
                    #if out_featimg[band, row, col] > 50 or out_featimg[band, row, col]< 0.001 : print(band, row, col, out_featimg[band, row, col])

    return out_featimg

@njit(nogil= True)
def GetBilinearElement(featimg, classimg, row, col):

    # In : feature img, label img, current row and col
    # out : the list of row, col and pixvalue for the founded bilinear candidate
    
    imgH = featimg.shape[0]
    imgW = featimg.shape[1]

    if row != 0 or row != (imgH - 1) or col != 0 or col != (imgW - 1):
        UL = FindnotMissingValue(classimg, row-1, col-1, plus = False, direction = True)
        UR = FindnotMissingValue(classimg, row-1, col+1, plus = True, direction = True)
        BL = FindnotMissingValue(classimg, row+1, col-1, plus = False, direction = True)
        BR = FindnotMissingValue(classimg, row+1, col+1, plus = True, direction = True)
    elif (row == 0 or row == (imgH - 1) ) and ( col != 0 or col != (imgW - 1) ):
        UL = FindnotMissingValue(classimg, row-1, col-1, plus = False, direction = False)
        UR = FindnotMissingValue(classimg, row-1, col+1, plus = False, direction = False)
        BL = FindnotMissingValue(classimg, row+1, col-1, plus = True, direction = False)
        BR = FindnotMissingValue(classimg, row+1, col+1, plus = True, direction = False)
    elif (row != 0 or row != (imgH - 1) ) and ( col == 0 or col == (imgW - 1) ):
        UL = FindnotMissingValue(classimg, row-1, col-1, plus = False, direction = True)
        UR = FindnotMissingValue(classimg, row-1, col+1, plus = True, direction = True)
        BL = FindnotMissingValue(classimg, row+1, col-1, plus = False, direction = True)
        BR = FindnotMissingValue(classimg, row+1, col+1, plus = True, direction = True)
    else:
        UL = FindnotMissingValue(classimg, row-1, col-1, plus = False, direction = True)
        UR = FindnotMissingValue(classimg, row-1, col+1, plus = True, direction = True)
        BL = FindnotMissingValue(classimg, row+1, col-1, plus = False, direction = True)
        BR = FindnotMissingValue(classimg, row+1, col+1, plus = True, direction = True)

    intp_row = np.array([UL[0], UR[0], BL[0], BR[0]])
    intp_col = np.array([UL[1], UR[1], BL[1], BR[1]])
    intp_pixvalue = np.array([0, 0, 0, 0], dtype=np.float32)
    for i in range(len(intp_row)):
        intp_pixvalue[i] = -1.0 if intp_row[i] == -1.0 else featimg[intp_row[i], intp_col[i]]
        #np.append(intp_pixvalue, pixValue)
    #intp_pixvalue = np.array([UL[2], UR[2], BL[2], BR[2]])

    return intp_row, intp_col, intp_pixvalue

@njit(nogil= True)
def FindnotMissingValue(classimg, row, col, plus = True, direction = True):
    # plus : T(plus) F(minus) # direction : T(horizonal), F(Vertical)
    # Find the pixel that not Missing value along the specific direction

    direct = np.array([0, 1]) if direction else np.array([1, 0])
    step_idx = 1 if plus else -1

    if row >= (classimg.shape[0] - 1) or row < 0 : return np.array([-1, -1])
    if col >= (classimg.shape[1] - 1) or col < 0 : return np.array([-1, -1])

    while(classimg[row, col] == 1):
        row = row + direct[0] * step_idx * 1
        col = col + direct[1] * step_idx * 1

        if row >= (classimg.shape[0] - 1) or row < 0 : return np.array([-1, -1])
        if col >= (classimg.shape[1] - 1) or col < 0 : return np.array([-1, -1])

    return np.array([row, col])

@njit(nogil= True)
def CheckMinus(valueA, valueB):
    
    out = np.array([-999.0])

    if valueA > 0 : out = np.append(out, valueA)
    if valueB > 0 : out = np.append(out, valueB)
    #out = np.delete(out, 0)
    
    return out
    
@njit(nogil= True)
def BilinearInterpolation(row, col, rowList, colList, pixValueList):
    
    # UL, UR, BL, BR
    y1Check = np.delete(CheckMinus(pixValueList[0], pixValueList[1]), 0)
    y2Check = np.delete(CheckMinus(pixValueList[2], pixValueList[3]), 0)

    if len(y1Check) == 2 and colList[1] != colList[0]: 
        y1 = pixValueList[0] * ((colList[1] - col)/(colList[1] - colList[0])) + pixValueList[1] * ((col - colList[0])/(colList[1] - colList[0]))
    elif len(y1Check) == 1 :#or colList[1] == colList[0]:
        y1 = y1Check[0]
    else : y1 = -1
    
    if len(y2Check) == 2 and colList[3] != colList[2]: 
        y2 = pixValueList[2] * ((colList[3] - col)/(colList[3] - colList[2])) + pixValueList[3] * ((col - colList[2])/(colList[3] - colList[2]))
    elif len(y2Check) == 1 :#or colList[3] == colList[2]:
        y2 = y2Check[0]
    else : y2 = -1
        
    if y1 != -1 and y2 != -1 and (rowList[2] or rowList[0]) == -1 : intpPixvalue = (y1 + y2)/2
    elif y1 != -1 and y2 != -1 :#and rowList[2] != rowList[0]:
        intpPixvalue = y1 * ((rowList[2] - row)/(rowList[2] - rowList[0])) + y2 * ((row - rowList[0])/(rowList[2] - rowList[0]))
    elif y1 == -1 and y2 != -1 : intpPixvalue = y2
    elif y1 != -1 and y2 == -1 : intpPixvalue = y1
    else : intpPixvalue = 0

    return intpPixvalue