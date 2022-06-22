import laspy
import numpy as np
from numba import njit
from numba import njit, prange

def ReadLAZ(path):

    laz = laspy.read(path)
    print("Point Cloud Sucessfully Read")

    return laz

def WriteLAZ(laz, xp, yp, zp, classification, name):

    header = laz.header
    laz_out = laspy.create(point_format=header.point_format, file_version=header.version)
    laz_out.x = xp 
    laz_out.y = yp
    laz_out.z = zp
    laz_out.classification = classification
    laz_out.write(name)

    return laz_out

def WriteLAZ_int(laz, xp, yp, zp, intensity, classification, name):

    header = laz.header
    laz_out = laspy.create(point_format=header.point_format, file_version=header.version)
    laz_out.x = xp 
    laz_out.y = yp
    laz_out.z = zp
    laz_out.intensity = intensity
    laz_out.classification = classification
    laz_out.write(name)

    return laz_out
    
@njit(nogil= True)
def GetMinMax(xp):

    xmax = xp.max()
    xmin = xp.min()
    
    return xmax, xmin

def GetLAZInfo(laz):

    xp = np.array(laz.x)
    yp = np.array(laz.y)
    zp = np.array(laz.z)
    intensity = np.array(laz.intensity)
    classification = np.array(laz.classification)
    print("Info Read")

    return xp, yp, zp, intensity, classification

def GetLAZInfo_noInt(laz):

    xp = np.array(laz.x)
    yp = np.array(laz.y)
    zp = np.array(laz.z)
    classification = np.array(laz.classification)
    print("Info Read")

    return xp, yp, zp, classification

def GetImgProjectionInfo(xp, yp, zp):
    
    xmax, xmin = GetMinMax(xp)
    ymax, ymin = GetMinMax(yp)
    zmax, zmin = GetMinMax(zp)
    maxmin = np.array([xmax, ymax, zmax, xmin, ymin, zmin])

    imgW = round(xmax - xmin)
    imgH = round(ymax - ymin)

    return imgW, imgH, maxmin

@njit(nogil= True)
def Normalization(zp, zmax, zmin):
    
    zp = zp - zmin
    zp = zp / (zmax - zmin)

    return zp

@njit(nogil= True)
def PointCloudShift(xp, yp, xmin, ymin):

    shift_xp = xp - xmin
    shift_yp = yp - ymin

    return shift_xp, shift_yp

## LAZ information : Class = 0-255
## Obtain the class that owns more than 1 point
def GetAllClass(classification):

    AllClass = []

    for i in range(256):
        if(len(classification[classification == i]) != 0):
            AllClass.append(i)

    AllClass = np.array(AllClass)

    return AllClass

@njit(nogil= True, parallel = False)
def ClassifyPointsTW(xp, yp, zp, minImg, classimg, classification, xmin, ymin, thres, sigma):
    CF_Matrix = np.zeros(4) #GG, G  #TP, TN, FP, FN
    notice = "#TP, TN, FP, FN \n " + str(len(classification[classification==2])) + "  " + str(len(classification[classification==31])) + "\n"
    
    imgH = minImg.shape[1] - 1

    for i in range(len(xp)):            
        cl_point = classification[i]
        # classification_out = np.zeros(len(classification))
        # If the point is predicted as a ground point and located within the std, 
        # classify as the ground points.

        if classimg[imgH - int(yp[i] - ymin), int(xp[i] - xmin)] == 2: # IsGround 

            # If InPrecision
            if (zp[i] - minImg[0, imgH - int(yp[i] - ymin), int(xp[i] - xmin)]) <= thres * sigma : # Positive

                if cl_point == 0 or cl_point == 1 or cl_point == 2 : CF_Matrix[0] += 1 # TP
                else : CF_Matrix[2] += 1 # FP
                
                classification[i] = 2 #Ground

            # If not InPrecision    
            else : # Negative
                
                if cl_point == 0 or cl_point == 1 or cl_point == 2 : CF_Matrix[3] += 1 # FN
                else : CF_Matrix[1] += 1 # TN

                classification[i] = 31 # Non-Ground

        elif classimg[imgH - int(yp[i] - ymin), int(xp[i] - xmin)] == 0: # not IsGround 

            if cl_point == 0 or cl_point == 1 or cl_point == 2 : CF_Matrix[3] += 1 # FN
            else : CF_Matrix[1] += 1 # TN

            classification[i] = 31 # Non-Ground

        else : # Classified to Missing Value # Negative

            if cl_point == 0 or cl_point == 1 or cl_point == 2 : CF_Matrix[3] += 1 # FN 
            else : CF_Matrix[3] += 1 # FN

            classification[i] = 31 #Non-Ground

    return classification, CF_Matrix, notice


@njit(nogil= True, parallel = False)
def ClassifyPoints(xp, yp, zp, minImg, classimg, classification, xmin, ymin, thres, sigma):
    CF_Matrix = np.zeros(9) # (Pr: a(row), GT:b(col))(G, NG MV)： 
                                # 1:GG  2:NGG  3:MVG 
                                # 4:GNG 5:NGNG 6:MVNG 
                                # 7:GMV 8:NGMV 9:MVMV <- No these Kind

    notice = "# (Pr: a(row), GT:b(col))(G, NG MV)： \n # 1:GG  2:NGG  3:MVG \n # 4:GNG 5:NGNG 6:MVNG \n # 7:GMV 8:NGMV 9:MVMV <- No these Kind\n"
                                    
    for i in range(len(xp)):            
        cl_point = classification[i]
        # If the point is predicted as a ground point and located within the std, 
        # classify as the ground points.
        if classimg[int(yp[i] - ymin), int(xp[i] - xmin)] == 0: 

            if (zp[i] - minImg[0, int(yp[i] - ymin), int(xp[i] - xmin)]) <= thres * sigma :

                if cl_point == 0 or cl_point == 1 or cl_point == 2 : CF_Matrix[0] += 1 # 1
                else : CF_Matrix[3] += 1 # 4
                
                classification[i] = 0
                
            else : 
                
                if cl_point == 0 or cl_point == 1 or cl_point == 2 : CF_Matrix[1] += 1 # 2
                else : CF_Matrix[4] += 1 # 5

                classification[i] = 1

        elif classimg[int(yp[i] - ymin), int(xp[i] - xmin)] == 1:

            if cl_point == 0 or cl_point == 1 or cl_point == 2 : CF_Matrix[1] += 1 # 2
            else : CF_Matrix[4] += 1 # 5

            classification[i] = 1

        else :

            if cl_point == 0 or cl_point == 1 or cl_point == 2 : CF_Matrix[2] += 1 # 3
            else : CF_Matrix[5] += 1 # 6

            classification[i] = 2    

    return classification, CF_Matrix, notice

@njit(nogil= True, parallel = False)
def ClassifyPoints_NonZero(xp, yp, zp, minImg, classimg, classification, xmin, ymin, thres, sigma):
    CF_Matrix = np.zeros(4) #TP, TN, FP, FN
    notice = "#TP, TN, FP, FN \n" 

    for i in range(len(xp)):
        # If the point is predicted as a ground point and located within the std, 
        # classify as the ground points.
        cl_point = classification[i]
        # Check if ground
        if classimg[int(yp[i] - ymin), int(xp[i] - xmin)] == 1: 
            #Ground
            if (zp[i] - minImg[0, int(yp[i] - ymin), int(xp[i] - xmin)]) <= thres * sigma :
                
                if cl_point == 0 or cl_point == 1 or cl_point == 2 : CF_Matrix[0] += 1 # TP
                else : CF_Matrix[2] += 1 # FP

                classification[i] = 1

            #Non-Ground
            else : 
                if cl_point == 0 or cl_point == 1 or cl_point == 2 : CF_Matrix[3] += 1 # FN
                else : CF_Matrix[1] += 1 # TN

                classification[i] = 2
        
        # Check if Non-Ground
        elif classimg[int(yp[i] - ymin), int(xp[i] - xmin)] == 2:

            if cl_point == 0 or cl_point == 1 or cl_point == 2 : CF_Matrix[3] += 1 # FN
            else : CF_Matrix[1] += 1 # TN

            classification[i] = 2
        
        # Check if Missing Value(No this class)
        #else : classification[i] = 0

    return classification, CF_Matrix, notice