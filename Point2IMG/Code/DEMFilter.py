from math import sqrt
import ARSEMLazTW as ARSEMLaz
import numpy as np
from os.path import basename
from numba import njit
import glob

LAZhome = "D:\Point2IMG\Taiwan\Taiwan_Point_Cloud/103_plane\pointcloud"

#lazname = "D:\Point2IMG\Taiwan\Taiwan_Point_Cloud/103_plane\pointcloud/97224090.las"
#lazname = "D:\Point2IMG\Taiwan\Taiwan_Point_Cloud/103_plane\pointcloud/Building/96221008.las"
lazname = "D:\Point2IMG\Taiwan\Taiwan_Point_Cloud/103_plane\pointcloud/Building/95183019.las"
#lazname = "D:\Point2IMG\Taiwan\Taiwan_Point_Cloud/103_plane\pointcloud/96221024.las"
out_LAZname = "D:\Point2IMG\Taiwan\TestFilter_0502/" + basename(lazname)[0:-4] + "_reclas_1_iter.las"

@njit(nogil= True)
def Reliable_Ground_points(xp, yp, zp, classification, imgW, imgH):

    # Since the Original TW point cloud owns reliable ground
    # Use the reliable ground points to get the reliableelevation
    # Input : xp, yp, zp, classification, Output: ground points within every 1m * 1m grid

    GPt_Container = np.zeros((10, imgH, imgW))

    for i in range(len(xp)):
        # Throw the ground point into the container to the non-zero position in the grid
        if classification[i] == 2 :
        
            zero_idx = 0
        
            while(zero_idx < GPt_Container.shape[0] and GPt_Container[zero_idx, int(yp[i]), int(xp[i])] != 0):
                zero_idx += 1

            if zero_idx < GPt_Container.shape[0]:
                GPt_Container[zero_idx, int(yp[i]), int(xp[i])] = zp[i]

    return GPt_Container

@njit(nogil= True)
def Elev_adjustment(imgW, imgH, GPt_Container):
    #LSA to caculate the estimated height
    adj_elev_img = np.zeros((2, imgH, imgW))

    for i in range(imgH):
        for j in range(imgW):
            elevs = GPt_Container[:, i, j][GPt_Container[:, i, j] != 0 ]

            if len(elevs) < 1: continue
            elif len(elevs) == 1 :
                adj_elev_img[0, i, j] = elevs[0]
                adj_elev_img[1, i, j] = 0
                
            else :
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

@njit(nogil= True)
def Possible_Ground_points(xp, yp, zp, imgW, imgH, adj_elev_img, prior = 0.048):

    # Grabbing the points by considering  
    # the estimated elevation and the observation prior

    GPt_Container = np.zeros((15, imgH, imgW))

    for i in range(len(zp)):
        
        est_z = adj_elev_img[0, int(yp[i]), int(xp[i])]

        if est_z != 0 and (zp[i] < est_z + 2 * prior) and (zp[i] > est_z - 2 * prior) :

            zero_idx = 0

            while(zero_idx < GPt_Container.shape[0] and GPt_Container[zero_idx, int(yp[i]), int(xp[i])] != 0):
                zero_idx += 1

            if zero_idx < GPt_Container.shape[0] :
                GPt_Container[zero_idx, int(yp[i]), int(xp[i])] = zp[i]

    return GPt_Container

@njit(nogil= True)
def Missing_Value_Intep(adj_elev_img):
    #Interpolate the missing value
    for i in range(1, adj_elev_img.shape[1] - 1):
        for j in range(1, adj_elev_img.shape[2] - 1):

            if adj_elev_img[0, i, j] == 0:
                pt_list = np.array([adj_elev_img[0, i + 1, j], adj_elev_img[0, i - 1, j], 
                                adj_elev_img[0, i, j + 1] != 0, adj_elev_img[0, i, j - 1]])
                std_list = np.array([adj_elev_img[1, i + 1, j], adj_elev_img[1, i - 1, j], 
                                adj_elev_img[1, i, j + 1] != 0, adj_elev_img[1, i, j - 1]])
                dividecount = 4
                for p in pt_list:
                    if p == 0: dividecount -= 1

                #if adj_elev_img[0, i + 1, j] != 0 and adj_elev_img[0, i - 1, j] != 0 and adj_elev_img[0, i, j + 1] != 0 and adj_elev_img[0, i, j - 1] != 0 :
                adj_elev_img[0, i, j] = (pt_list.sum()) / (dividecount + 0.00000000001)
                adj_elev_img[1, i, j] = ((std_list[0]**2 + std_list[1]**2 + std_list[2]**2 + std_list[3]**2)**0.5) / sqrt(dividecount + 0.00000000001)
    
    return adj_elev_img

@njit(nogil= True)
def Ground_Points_Classify(xp, yp, zp, imgW, imgH, classification, adj_elev_img):

    #The point may be re-classified as ground points if
    # 1. It's elevation locates at 2 * sigma0
    # 2. If the point is the only point within the grid:
    #      Caculate it's average elevation and propogated std by it's 4 neighbours

    to_ground_count = 0

    for i in range(len(zp)):

        if classification[i] != 2:
            upper_thre = adj_elev_img[0, int(yp[i]), int(xp[i])] + 2 * adj_elev_img[1, int(yp[i]), int(xp[i])]
            lower_thre = adj_elev_img[0, int(yp[i]), int(xp[i])] - 2 * adj_elev_img[1, int(yp[i]), int(xp[i])]
            
            # Case 1 : More than 1 ground point in the grid
            if (zp[i] <= upper_thre and zp[i] >= lower_thre) :
                classification[i] = 2
                to_ground_count += 1
            
            # Case 2 : 0 ground point in the grid, non-ground only 
            #       => obtain the value from 4 neighbour
            #          if the elev == 0, the elev will not be taken into account            
            if ( 0 < int(yp[i]) < (imgH - 1) ) and ( 0 < int(xp[i]) < (imgW - 1) ) : #Ignore the border case
                pt_list = np.array([adj_elev_img[0, int(yp[i]) + 1, int(xp[i])], adj_elev_img[0, int(yp[i]) - 1, int(xp[i])], 
                                adj_elev_img[0, int(yp[i]), int(xp[i]) + 1], adj_elev_img[0, int(yp[i]), int(xp[i]) - 1]])
                std_list = np.array([adj_elev_img[1, int(yp[i]) + 1, int(xp[i])], adj_elev_img[1, int(yp[i]) - 1, int(xp[i])], 
                                 adj_elev_img[1, int(yp[i]), int(xp[i]) + 1], adj_elev_img[1, int(yp[i]), int(xp[i]) - 1]])
                
                dividecount = 4
                for p in pt_list:
                    if p == 0 : dividecount -= 1

                avgelev = (pt_list[0] + pt_list[1] + pt_list[2] + pt_list[3]) / (dividecount + 0.00000000001)
                avgstd = ((std_list[0]**2 + std_list[1]**2 + std_list[2]**2 + std_list[3]**2)**0.5)/(sqrt(dividecount)+ 0.00000000001)

                if (avgelev - 2 * avgstd) < zp[i] < (avgelev + 2 * avgstd):
                    classification[i] = 2
                    to_ground_count += 1

    print(str(to_ground_count) + " of points have been transed")

    return classification

def LasNoiseFilter(laz, out_LAZname, prior = 0.048):

    Ep, Np, Hp, intensity, classification = ARSEMLaz.GetLAZInfo(laz)
    imgW, imgH, xymaxmin= ARSEMLaz.GetImgProjectionInfo(Ep, Np, Hp)
    Ep, Np = ARSEMLaz.PointCloudShift(Ep, Np, xymaxmin[3], xymaxmin[4]) #[Emax, Nmax, Hmax, Emin, Nmin, Hmin]

    GP_Container_Actual = Reliable_Ground_points(Ep, Np, Hp, classification, imgW, imgH)        
    Adj_elev_img = Elev_adjustment(imgW, imgH, GP_Container_Actual)
    GPt_Container_possible = Possible_Ground_points(Ep, Np, Hp, imgW, imgH, Adj_elev_img, prior = prior)
    Adj_elev_img_possible = Elev_adjustment(imgW, imgH, GPt_Container_possible)
    import ARSEMimgTW
    ARSEMimgTW.SaveTiff_1inp("out.tif", Adj_elev_img_possible, nodata=-10)

    classification = Ground_Points_Classify(Ep, Np, Hp, imgW, imgH, classification, Adj_elev_img_possible)

    print("Writing New Las")
    laz = ARSEMLaz.WriteLAZ_int(laz, Ep + xymaxmin[3], Np + xymaxmin[4], Hp, intensity, classification, out_LAZname)

    return laz


if __name__ == "__main__":

    laz = ARSEMLaz.ReadLAZ(lazname)
    LasNoiseFilter(laz, "D:\Point2IMG\Taiwan\Taiwan_Point_Cloud/103_plane/" + basename(lazname)[0:-4] + "_reclass.las")
    
    Ep, Np, Hp, classification = ARSEMLaz.GetLAZInfo_noInt(laz)
    imgW, imgH, xymaxmin= ARSEMLaz.GetImgProjectionInfo(Ep, Np, Hp)
    Ep, Np = ARSEMLaz.PointCloudShift(Ep, Np, xymaxmin[3], xymaxmin[4]) #[Emax, Nmax, Hmax, Emin, Nmin, Hmin]

    GP_Container_Actual = Reliable_Ground_points(Ep, Np, Hp, classification, imgW, imgH)        
    Adj_elev_img = Elev_adjustment(GP_Container_Actual)
    GPt_Container_possible = Possible_Ground_points(Ep, Np, Hp, Adj_elev_img)
    Adj_elev_img_possible = Elev_adjustment(GPt_Container_possible)
    classification = Ground_Points_Classify(Ep, Np, Hp, imgW, imgH, classification, Adj_elev_img_possible)

    laz = ARSEMLaz.WriteLAZ(laz, Ep + xymaxmin[3], Np + xymaxmin[4], Hp, classification, out_LAZname)