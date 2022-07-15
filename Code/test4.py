import ARSEMimgTW
import AfterProcessingTW
from keras_segmentation_mod.predict import featImgCropper
from os.path import basename
import numpy as np

featimg = "D:/Point2IMG/Taiwan/0707_TestClip/UnClipped_Img/94181061_DEMfilt.tif"
out = 'D:\Point2IMG\Taiwan/0707_TestClip/out/'
#labelimg = "D:/Point2IMG/Taiwan/0707_TestClip/UnClipped_Png/94181061_DEMfilt.png"
#outImgName = basename(labelimg)[0:-4]

featimg = ARSEMimgTW.ReadImg(featimg)
#labelimg = ARSEMimgTW.ReadImg(labelimg)

locationlist, croppedList, imgInfo = featImgCropper(featimg, 256, 256, 128)

# for i, tif in enumerate(croppedList):
#     ARSEMimgTW.SaveTiff_1inp(out + '_' + str(locationlist[0, i]) + '_' + str(locationlist[1, i]) + '.tif', tif, 0)

# print(croppedList)
# print(locationlist)

def featImgMerger(featTIFList, locationList, imgInfo):
    out = np.zeros((imgInfo[0], imgInfo[1], imgInfo[2]))
    #imgInfo：[0:b, 1:H, 2:W, 3:cropHcount, 4:cropWcount, 5:stride, 6:cropH, 7cropW]
    imgH = imgInfo[1]
    imgW = imgInfo[2]
    cropHcount = imgInfo[3]
    cropWcount = imgInfo[4]
    stride = imgInfo[5]
    cropH = imgInfo[6]
    cropW = imgInfo[7]

    for i, img in enumerate(featTIFList):

        h = int(locationList[0, i])
        w = int(locationList[1, i])
        
        if h == cropHcount and w == cropWcount: # Corner
            out[:, imgH - 1 - cropH :  imgH - 1, imgW - 1 - cropW : imgW - 1] = img
        elif h == cropHcount and w != cropWcount: # Horizonal Bottom
            out[:, imgH - 1 - cropH : imgH - 1, w * stride : w * stride + cropW] = img
        elif h != cropHcount and w == cropWcount: # Vertical RSide
            out[:, h * stride : (h * stride + cropH) , (imgW - 1 - cropW) : (imgW - 1)] = img
        elif h == 0 and w != cropWcount: # Horizonal Top
            out[:, 0 : cropH, w * stride : w * stride + cropW] = img
        elif h != cropHcount and w == 0: # Vertical LSide
            out[:, h * stride : h * stride + cropH, 0 : cropW] = img
        else : #Middle
            middle = img[:, int(cropH * 0.25) : int(cropH * 0.75), int(cropW * 0.25) : int(cropW * 0.75)]
            out[:, h * stride + int(cropH * 0.25) : h * stride + + int(cropH * 0.75), w * stride + int(cropW * 0.25) : w * stride + int(cropW * 0.75)] = middle
        
    return out

feat = featImgMerger(croppedList, locationlist, imgInfo)
ARSEMimgTW.SaveTiff_1inp(out + 'feat' + '.tif', feat, 0)
#from keras.utils import multi_gpu_utils