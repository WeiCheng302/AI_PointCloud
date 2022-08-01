import ARSEMimgTW
import AfterProcessingTW
from keras_segmentation_mod.predict import featImgCropper
from os.path import basename
import numpy as np

featimg = "D:/Point2IMG/Taiwan/0707_TestClip/UnClipped_Img/94181061_DEMfilt.tif"
out = 'D:\Point2IMG\Taiwan/0707_TestClip/try/'
#labelimg = "D:/Point2IMG/Taiwan/0707_TestClip/UnClipped_Png/94181061_DEMfilt.png"
#outImgName = basename(labelimg)[0:-4]

featimg = ARSEMimgTW.ReadImg(featimg)

for i in range(100):
    tif = featimg[:, 1700 + i : 1956 + i, 0:256]
    ARSEMimgTW.SaveTiff_1inp(out + 'feat' + str(i) + '.tif', tif, 0)
