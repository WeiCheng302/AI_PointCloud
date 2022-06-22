import ARSEMimgTW

featimg = "D:\Point2IMG\Taiwan/0602_Meanfill/UnClipped_Img_All/94182060_DEMfilt.tif"
labelimg = "D:\Point2IMG\Taiwan/0602_Meanfill/UnClipped_Png_All/94182060_DEMfilt.png"
outfeatimg = "D:\Point2IMG\Taiwan/0602_Meanfill/94182060_DEMfilt_intp2.tif"

featimg = ARSEMimgTW.ReadImg(featimg)
labelimg = ARSEMimgTW.ReadImg(labelimg)
out = ARSEMimgTW.AdaptiveBilinearInterpolation(featimg, labelimg)

ARSEMimgTW.SaveTiff_1inp(outfeatimg, out, -1)