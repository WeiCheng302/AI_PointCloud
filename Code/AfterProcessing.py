import numpy as np
import glob
import os
import ARSEMimg
import ARSEMLaz
from laspy import read
#from BackProject import GetMinMax, CreateMinImg, ClassifyPoints, digit
from os.path import basename
from ARSEMimg import ReadImg, CreateImgArray, SetImgValueAll, SavePNG

filename = "C_25DN2"
#filename = "C_39AZ2"
#filename = "C_31HZ1"
#filename = "C_57FN1"

#PointCloudName = "D:/Point2IMG/Point/" + filename + ".LAZ"
#motherfolder = "D:/Point2IMG/Original_Img_0407_2/"
motherfolder = "D:/Point2IMG/Original_Img_0331/"


Non_Ground_path = motherfolder + filename + "_Non_Ground.txt"
Ground_path = motherfolder + filename + "_Ground.txt"

#checkpoint = "D:/image-segmentation-keras/0325/OutPut_0325\_0325"
#checkpoint = "D:/image-segmentation-keras/0329/CheckPoints\ckp_"
checkpoint = "D:/image-segmentation-keras/_0331/CheckPoints\_0331"
inpt_allimg = motherfolder + "All_Img"
inpt_allLabel = motherfolder + "All_Label"

inpt_trainimg = motherfolder + "Train_Img"
inpt_trainLabel = motherfolder + "Train_Label"

inpt_testimg = motherfolder + "Test_Img"
inpt_testLabel = motherfolder + "Test_Label"

out_merge = motherfolder
out_predict_folder = motherfolder + "Predicted_Label4"

thres = 2
sigma = 0.05

def digit(a):
    return str("{:.3f}".format(a))

def commander(Predict = False, Eval_All = False, Eval_Train = False, Eval_Test = False, Eval_All_Ignore = False):
    
    DirCurrent = os.getcwd()
    os.chdir("D:\image-segmentation-keras")

    #Prediction
    if Predict :
        os.system("python -m keras_segmentation predict --checkpoints_path=" + checkpoint + " --input_path=" + inpt_allimg + " --output_path=" + out_predict_folder)
    
    #Evaluation All
    if Eval_All :
        os.system("python -m keras_segmentation evaluate_model --checkpoints_path=" + checkpoint +" --images_path=" + inpt_allimg + " --segs_path=" + inpt_allLabel)
    
    if Eval_All_Ignore :
        os.system("python -m keras_segmentation evaluate_model_ignore --checkpoints_path=" + checkpoint +" --images_path=" + inpt_allimg + " --segs_path=" + inpt_allLabel)

    #Evaluation Train
    if Eval_Train :
        os.system("python -m keras_segmentation evaluate_model --checkpoints_path=" + checkpoint +" --images_path=" + inpt_trainimg + " --segs_path=" + inpt_trainLabel)
    
    #Evaluation Test
    if Eval_Test :
        os.system("python -m keras_segmentation evaluate_model --checkpoints_path=" + checkpoint +" --images_path=" + inpt_testimg + " --segs_path=" + inpt_testLabel)
    
    os.chdir(DirCurrent)
    print("Commander Finished")

def CreateMask(imgfile):
    mask = ARSEMimg.ReadImg(inpt_allimg + "/" + basename(imgfile)[0:-4] + ".tif")
    
    mask[mask == -32768] = 0
    mask[mask != -32768] = 1

    return np.array(mask)

def MergeImg(filename, out_predict_folder, out_merge, ignore_0 = False):

    imgList = []
    outImg = ARSEMimg.CreateImgArray(6250, 5000, 1, np.uint8)

    PredictImgList = glob.glob(out_predict_folder + "/*.png")

    for imgname in PredictImgList:  # Obtaining the C_XXXXXX file name
        if basename(imgname)[0:7] == filename :
            imgList.append(imgname)
    
    for imgfile in imgList: # Set the image value
        img = ARSEMimg.ReadImg(imgfile)        

        imgH = int(img.shape[0])
        imgW = int(img.shape[1])
        h = int(basename(imgfile)[0:-4].split('_')[2])
        w = int(basename(imgfile)[0:-4].split('_')[3])
            
        if ignore_0:
            mask = CreateMask(imgfile)
            outImg = ARSEMimg.SetImgValueAll(mask[0] * img, outImg, imgH, imgW, h, w)
            #print(imgfile)
        else :
            outImg = ARSEMimg.SetImgValueAll(img, outImg, imgH, imgW, h, w)

    SavePNG(out_merge + basename(filename) + '_Predict3.png', outImg[0])

    return outImg

def MergeImg_BCE_4Band(filename, out_predict_folder, out_merge, ignore_0 = False):

    imgList = []
    outImg = ARSEMimg.CreateImgArray(6250, 5000, 1, np.float32)

    PredictImgList = glob.glob(out_predict_folder + "/*.tif")

    for imgname in PredictImgList:  # Obtaining the C_XXXXXX file name
        if basename(imgname)[0:7] == filename :
            imgList.append(imgname)
    
    for imgfile in imgList: # Set the image value
        img = ARSEMimg.ReadImg(imgfile)        

        imgH = int(img.shape[0])
        imgW = int(img.shape[1])
        h = int(basename(imgfile)[0:-4].split('_')[2])
        w = int(basename(imgfile)[0:-4].split('_')[3])
            
        if ignore_0:
            mask = CreateMask(imgfile)
            outImg = ARSEMimg.SetImgValueAll(mask[0] * img, outImg, imgH, imgW, h, w)
            #print(imgfile)
        else :
            outImg = ARSEMimg.SetImgValueAll(img, outImg, imgH, imgW, h, w)

    SavePNG(out_merge + basename(filename) + '_Predict4.png', outImg[0])

    return outImg


# Remember to modify the net structurt and the activate function before evaluating


def precision_idx(CF_Matrix, dim):

    if dim == 9:
        accu = (CF_Matrix[0]  +CF_Matrix[4])/CF_Matrix.sum()
        prec_g = CF_Matrix[0]/(CF_Matrix[0]+CF_Matrix[3])
        prec_ng = CF_Matrix[4]/(CF_Matrix[1]+CF_Matrix[4])
        Rec_g = CF_Matrix[0]/(CF_Matrix[0]+CF_Matrix[1])
        Rec_ng = CF_Matrix[4]/(CF_Matrix[3]+CF_Matrix[4])
        F1_g = str((2 * prec_g * Rec_g) / (prec_g + Rec_g))
        F1_ng = str((2 * prec_ng * Rec_ng) / (prec_ng + Rec_ng))
        kap_g = str((prec_g - accu)/(1 - accu))
        kap_ng = str((prec_ng - accu)/(1 - accu))

        precision_idx = ("Accuracy = " + str(accu) + "\n" 
                        "Precision_Ground = " + str(prec_g) + "\n" + 
                        "Precision_NonGround = " + str(prec_ng) + "\n" 
                        "Recall_Ground = " + str(Rec_g) + "\n" + 
                        "Recall_NonGround = " + str(Rec_ng) + "\n"
                        "F1_Ground = " + str(F1_g) + "\n" + 
                        "F1_NonGround = " + str(F1_ng) + "\n"
                        "Kappa_Ground = " + str(kap_g) + "\n"
                        "Kappa_NonGround = " + str(kap_ng) + "\n")
    elif dim == 4:
        accu = (CF_Matrix[0]+CF_Matrix[3])/CF_Matrix.sum
        prec_g = CF_Matrix[0]/(CF_Matrix[0]+CF_Matrix[2])
        prec_ng = CF_Matrix[3]/(CF_Matrix[1]+CF_Matrix[3])
        Rec_g = CF_Matrix[0]/(CF_Matrix[0]+CF_Matrix[1])
        Rec_ng = CF_Matrix[3]/(CF_Matrix[2]+CF_Matrix[3])
        F1_g = str((2 * prec_g * Rec_g) / (prec_g + Rec_g))
        F1_ng = str((2 * prec_ng * Rec_ng) / (prec_ng + Rec_ng))
        kap_g = str((accu - prec_g)/(1 - prec_g))
        kap_ng = str((accu - prec_ng)/(1-prec_ng))
    
        precision_idx = ("Accuracy = " + str(accu) + "\n" 
                         "Precision_Ground = " + str(prec_g) + "\n" + 
                         "Precision_NonGround = " + str(prec_ng) + "\n" 
                         "Recall_Ground = " + str(Rec_g) + "\n" + 
                         "Recall_NonGround = " + str(Rec_ng) + "\n"
                        "F1_Ground = " + str(F1_g) + "\n" + 
                        "F1_NonGround = " + str(F1_ng) + "\n"
                        "Kappa_Ground = " + str(kap_g) + "\n"
                        "Kappa_NonGround = " + str(kap_ng) + "\n")
    
    else : print("No this Dimension")

    return precision_idx

def BackProjection(filename, outImg, ignore_Zero = False):

    PointCloudName = "D:/Point2IMG/Point/" + filename + ".LAZ"
    print("Reading Point Cloud " + PointCloudName)

    laz = read(PointCloudName)
    if type(outImg) == type("123"):
        classimg = np.array(ARSEMimg.ReadImg(outImg))
        #classimg = np.array(classimg)[0]
    else:
        classimg = outImg[0]

    print("Successfully Read The Point Cloud")   
    xp, yp, zp, classification = ARSEMLaz.GetLAZInfo_noInt(laz) 
    imgW, imgH ,xyMaxMin = ARSEMLaz.GetImgProjectionInfo(xp, yp)

    print("Start Caculating ")    
    
    minImg = ARSEMimg.CreateMinImg(xp, yp, zp, xyMaxMin[2], xyMaxMin[3], ARSEMimg.CreateImgArray(imgH, imgW, 1, np.float32) + 9999.0)
    if not ignore_Zero:
        classification, CF_Matrix, notice = ARSEMLaz.ClassifyPoints(xp, yp, zp, minImg, classimg, classification, xyMaxMin[2], xyMaxMin[3], thres, sigma)
        precision_index = precision_idx(CF_Matrix, 9)
    else:
        classification, CF_Matrix, notice = ARSEMLaz.ClassifyPoints_NonZero(xp, yp, zp, minImg, classimg, classification, xyMaxMin[2], xyMaxMin[3], thres, sigma)
        precision_index = precision_idx(CF_Matrix, 4)

    f = open(motherfolder + "ConfusionMatrix.txt", 'w')
    f.write(notice)
    f.writelines(precision_index)
    f.writelines(str(CF_Matrix))
    f.close()

    return xp, yp, zp, classification

if __name__ == "__main__":
    
    #Prediction
    commander(Predict = True, Eval_All = True, Eval_All_Ignore=False, Eval_Train = False, Eval_Test = False)

    #Merging Image    
    #outImg = MergeImg_BCE_4Band(filename, out_predict_folder, out_merge, ignore_0 = False)
    outImg = MergeImg(filename, out_predict_folder, out_merge, ignore_0 = True)


    #Back Projection
    #xp, yp, zp, classification = BackProjection(filename, outImg)
    
    #print("Start Writing")
    #f = open(Ground_path, 'w')
    #for i in range(len(xp)):
    #    if classification[i] == 1 :
    #        f.write(digit(xp[i]) + ' ' + digit(yp[i]) + ' ' + digit(zp[i]) + ' ' + str(classification[i]) + '\n')
    #    if i % 50000000 == 1000 : print(digit(i/len(xp)) + " of data has been processed ...")
    #f.close()

    ##f = open(Non_Ground_path, 'w')
    #for i in range(len(xp)):
    #    if classification[i] != 0 :
    #        f.write(digit(xp[i]) + ' ' + digit(yp[i]) + ' ' + digit(zp[i]) + ' ' + str(classification[i]) + '\n')
    #    if i % 50000000 == 1000 : print(digit(i/len(xp)) + " of data has been processed ...")
    #f.close()

    '''
    imgList = []
    outImg = ARSEMimg.CreateImgArray(6250, 5000, 1, np.uint8)

    for imgname in ImgFileList:  # Obtaining the C_XXXXXX file name
        if basename(imgname)[0:7] == filename :
            imgList.append(imgname)

    for imgfile in imgList: # Set the image value
        img = ARSEMimg.ReadImg(imgfile)

        imgH = int(img.shape[0])
        imgW = int(img.shape[1])
        h = int(basename(imgfile)[0:-4].split('_')[2])
        w = int(basename(imgfile)[0:-4].split('_')[3])
            
        outImg = ARSEMimg.SetImgValueAll(img, outImg, imgH, imgW, h, w)             

    SavePNG(out_merge + basename(filename) + '_Predict.png', outImg[0])
    '''
