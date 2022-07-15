import ARSEMimgTW as ARSEMimg

FeatImgIn = 'D:\Point2IMG\Taiwan/0523_Plane/Val_Img/95203004_DEMfilt.tif'
LabelImgIn = 'D:\Point2IMG\Taiwan/0523_Plane/Val_Label/95203004_DEMfilt.png'
FeatImgOut = 'D:\Point2IMG\Taiwan/0523_Plane/95203004_DEMfilt_R.tif'
LabelImgOut = 'D:\Point2IMG\Taiwan/0523_Plane/95203004_DEMfilt_R.png'

Feat = ARSEMimg.ReadImg(FeatImgIn)
Label = ARSEMimg.ReadImg(LabelImgIn)

oFeat, oLabel = ARSEMimg.Resample(Feat, Label)

ARSEMimg.SavePNG(LabelImgOut, oLabel)
ARSEMimg.SaveTiff_1inp(FeatImgOut, oFeat, -0.1)