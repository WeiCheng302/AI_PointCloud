import ARSEMimgTW
import AfterProcessingTW
from os.path import basename

featimg = "D:/Point2IMG/Taiwan/0707_TestClip/UnClipped_Img/94181061_DEMfilt.tif"
labelimg = "D:/Point2IMG/Taiwan/0707_TestClip/UnClipped_Png/94181061_DEMfilt.png"
outImgName = basename(labelimg)[0:-4]

featimg = ARSEMimgTW.ReadImg(featimg)
labelimg = ARSEMimgTW.ReadImg(labelimg)

TiffList, TiffIDList, BTiffList, BTiffIDList = ARSEMimgTW.ClipTif_Border_Overlap(featimg, 256, 256)
PngList, PngIDList, BPngList, BPngIDList = ARSEMimgTW.ClipPng_Border_Overlap(labelimg, 256, 256)

mother_folder = "D:/Point2IMG/Taiwan/0707_TestClip"

Folder_UnClipped = mother_folder + "/UnClipped_Img/"
Folder_UnClippedPng = mother_folder + "/UnClipped_Png/"

Folder_UnClippedLabel_All = mother_folder + "/UnClipped_Png_All/"
Folder_UnClippedtrain_All = mother_folder + "/UnClipped_Img_All/"

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
out_predict_folder = mother_folder + "/Predicted_Label"


bands = 3
print("Saving All Images")
for i in range(len(PngList)):
    ARSEMimgTW.SaveTiff_ninp(FolderTif_All + outImgName + TiffIDList[bands * i] + ".tif", TiffList, i, bands, 0)
    ARSEMimgTW.SavePNG(FolderPng_All + outImgName + PngIDList[i] + ".png", PngList[i])

for i in range(len(BPngIDList)):
    ARSEMimgTW.SaveTiff_ninp(FolderTif_All + outImgName + BTiffIDList[bands * i] + ".tif", BTiffList, i, bands, 0)
    ARSEMimgTW.SavePNG(FolderPng_All + outImgName + BPngIDList[i] + ".png", BPngList[i])
        
    ARSEMimgTW.SaveTiff_ninp(FolderTif_Border + outImgName + BTiffIDList[bands * i] + ".tif", BTiffList, i, bands, 0)
    ARSEMimgTW.SavePNG(FolderPng_Border + outImgName + BPngIDList[i] + ".png", BPngList[i])

AfterProcessingTW.MergeImg('94181061_DEMfilt.png', out_predict_folder, mother_folder + '/' )