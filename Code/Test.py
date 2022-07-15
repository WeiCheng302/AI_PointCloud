from requests import head
import ARSEMLaz
import laspy
import numpy as np
from numba import jit, njit, prange
import threading
import time
laz = "D:\Point2IMG\Taiwan\Taiwan_Point_Cloud/103_plane\pointcloud/96221008.las"
dem_elip = "D:\Point2IMG\Taiwan\Taiwan_Point_Cloud/103_plane\dem\DEMe96221008_-3zv-3_ok_NNp2.las"
start = time.time()
laz = ARSEMLaz.ReadLAZ(laz)

xp, yp, zp, classification = ARSEMLaz.GetLAZInfo_noInt(laz)
imgW, imgH , xymaxmin= ARSEMLaz.GetImgProjectionInfo(xp, yp, zp)
xp, yp = ARSEMLaz.PointCloudShift(xp, yp, xymaxmin[3], xymaxmin[4])

@njit(nogil= True)
def a():
    # Take the ground points
    elev_Img = np.zeros((10, imgH, imgW))

    for i in range(len(xp)):
        if classification[i] == 2 :
        
            zero_idx = 0
        
            while(zero_idx < elev_Img.shape[0] and elev_Img[zero_idx, int(yp[i]), int(xp[i])] != 0):
                zero_idx += 1

            if zero_idx < elev_Img.shape[0]:
                elev_Img[zero_idx, int(yp[i]), int(xp[i])] = zp[i]

    return elev_Img

elev_Img = a()

@njit(nogil= True)
def b(elev_Img):
    #LSA to caculate the estimated height
    adj_elev_img = np.zeros((2, imgH, imgW))

    for i in range(imgH):
        for j in range(imgW):
            elevs = elev_Img[:, i, j][elev_Img[:, i, j] != 0 ]

            if len(elevs) < 1: continue

            A = np.zeros(( len(elevs), 1)) + 1
            L = np.zeros(( len(elevs), 1)) 
            for k in range(len(elevs)):
                L[k] = elevs[k]
            X = np.linalg.inv(A.transpose().dot(A)).dot(A.transpose().dot(L))
            V = A.dot(X) - L
            sigma0 = ( (V.transpose().dot(V)) / (len(V) - len(X)) )**0.5

            adj_elev_img[0, i, j] = X[0, 0]
            adj_elev_img[1, i, j] = sigma0[0, 0]

    return adj_elev_img
        
adj_elev_img = b(elev_Img)

@njit(nogil= True)
def e(adj_elev_img, prior = 0.048):
    # Take All of the ground points

    elev_Img_all = np.zeros((15, imgH, imgW))

    for i in range(len(zp)):
        
        est_z = adj_elev_img[0, int(yp[i]), int(xp[i])]
        a = zp[i]

        if est_z != 0 and (zp[i] < est_z + 2 * prior) and (zp[i] > est_z - 2 * prior) :

            zero_idx = 0

            while(zero_idx < elev_Img_all.shape[0] and elev_Img_all[zero_idx, int(yp[i]), int(xp[i])] != 0):
                zero_idx += 1

            if zero_idx < elev_Img_all.shape[0] :
                elev_Img_all[zero_idx, int(yp[i]), int(xp[i])] = zp[i]
                #print(zp[i])

    return elev_Img_all

elev_Img_all = e(adj_elev_img)

adj_elev_img_all = b(elev_Img_all)

@njit(nogil= True)
def d(adj_elev_img):
    #Interpolate the missing value
    for i in range(1, adj_elev_img.shape[1] - 1):
        for j in range(1, adj_elev_img.shape[2] - 1):

            if adj_elev_img[0, i + 1, j] != 0 and adj_elev_img[0, i - 1, j] != 0 and adj_elev_img[0, i, j + 1] != 0 and adj_elev_img[0, i, j - 1] != 0 :
                adj_elev_img[0, i, j] = (adj_elev_img[0, i + 1, j] + adj_elev_img[0, i - 1, j] + adj_elev_img[0, i, j + 1] + adj_elev_img[0, i, j - 1]) / 4
                adj_elev_img[1, i, j] = (adj_elev_img[0, i + 1, j]**2 + adj_elev_img[0, i - 1, j]**2 + adj_elev_img[0, i, j + 1]**2 + adj_elev_img[0, i, j - 1]**2)**0.5
    
    return adj_elev_img
#d(adj_elev_img)

@njit(nogil= True)
def c(classification, adj_elev_img):
    #ground point adaption
    count = 0

    for i in range(len(zp)):
        if classification[i] != 2:
            upper_tre = adj_elev_img[0, int(yp[i]), int(xp[i])] + 2 * adj_elev_img[1, int(yp[i]), int(xp[i])]
            lower_tre = adj_elev_img[0, int(yp[i]), int(xp[i])] - 2 * adj_elev_img[1, int(yp[i]), int(xp[i])]
            
            if (zp[i] <= upper_tre and zp[i] >= lower_tre) :
                classification[i] = 2
                count += 1
            
            if ( 0 < int(yp[i]) < (imgH - 1) ) and ( 0 < int(xp[i]) < (imgW - 1) ) :
                avgelev = (adj_elev_img[0, int(yp[i]) + 1, int(xp[i])] + adj_elev_img[0, int(yp[i]) - 1, int(xp[i])]
                        + adj_elev_img[0, int(yp[i]), int(xp[i]) + 1] + adj_elev_img[0, int(yp[i]), int(xp[i]) - 1]) / 4
                avgstd = (adj_elev_img[1, int(yp[i]) + 1, int(xp[i])]**2 + adj_elev_img[1, int(yp[i]) - 1, int(xp[i])]**2
                        + adj_elev_img[1, int(yp[i]), int(xp[i]) + 1]**2 + adj_elev_img[1, int(yp[i]), int(xp[i]) - 1]**2)**0.5
                
                if (avgelev - 2 * avgstd) < zp[i] < (avgelev + 2 * avgstd):
                    classification[i] = 20
                    count += 1

    print(count)

c(classification, adj_elev_img_all)

def writeLAZ(laz, xp, yp, zp, classification, name):

    header = laz.header
    laz_out = laspy.create(point_format=header.point_format, file_version=header.version)
    laz_out.x = xp 
    laz_out.y = yp
    laz_out.z = zp
    laz_out.classification = classification
    laz_out.write(name)

name = "D:\Point2IMG\Taiwan\Taiwan_Point_Cloud/test2.las"
writeLAZ(laz, xp + xymaxmin[3], yp + xymaxmin[4], zp, classification, name)
print(time.time() - start)
def digit(a):
    return str("{:.3f}".format(a))

print("Start Writing")
threads = []

def totxt(k):
    f = open("D:\Point2IMG\Taiwan\Taiwan_Point_Cloud/test" + str(k) + "_real.txt", 'w')
    for i in range(len(xp)):
        f.write(digit(xp[i] + xymaxmin[3]) + ' ' + digit(yp[i] + xymaxmin[4]) + ' ' + digit(zp[i]) + ' ' + str(classification[i]) + '\n')
        if i % 5000000 == 1000 : print(digit(i/len(xp)) + " of data has been processed ...")
    f.close()

totxt(1)

print("Done. ")





