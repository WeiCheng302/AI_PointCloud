import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

from time import time
import numpy as np
import glob, os
import ARSEMimgTW as ARSEMimg
import ARSEMLazTW as ARSEMLaz
from laspy import read
from numba import njit
from os.path import basename
from ARSEMimgTW import ReadImg, CreateImgArray, SetImgValueAll, SavePNG
from keras_segmentation_mod.predict import predict_from_featImg_binary

motherfolder = "D:\Point2IMG\Taiwan/0707_Seamless/"

#checkpoint = "D:/image-segmentation-keras/0707_LowVeg/CheckPoints\_0706_TW_LowVeg_3060_3lyr_MBunet_BCE_Interpol_demPre_RD_Adam"
checkpoint = "D:/image-segmentation-keras/0707_City/CheckPoints\_0706_TW_City_3060_3lyr_MBunet_BCE_Interpol_demPre_RD_Adam"
#checkpoint = "D:/image-segmentation-keras/0707_Forest/CheckPoints\_0706_TW_Forest_3060_3lyr_MBunet_BCE_Interpol_demPre_RD_Adam"

inpt_allimg = motherfolder + "All_Img"
inpt_allLabel = motherfolder + "All_Label"

inpt_UnClippedtrain = motherfolder + "UnClipped_Img_All/"
inpt_UnClippedLabel = motherfolder + "UnClipped_Png_All/"

out_merge = motherfolder
out_predict_folder = motherfolder + "Predicted_Label"

thres = 2
sigma = 0.05

def digit(a):
    return str("{:.3f}".format(a))

def MergeImg(filename, out_predict_folder, out_merge):

    imgList = []
    lasname = basename(filename)[0:-4] #12345678_DEMfilt  [:8]#
    png = ARSEMimg.ReadImg(inpt_UnClippedLabel + lasname + ".png")
    outImg = ARSEMimg.CreateImgArray(png.shape[0], png.shape[1], 1, np.uint8)

    PredictImgList = glob.glob(out_predict_folder + "/*.png")

    for imgname in PredictImgList:  # Obtaining the C_XXXXXX file name
        if basename(imgname)[0:8] == lasname[0:8] :
            imgList.append(imgname)
    
    for imgfile in imgList: # Set the image value
        img = ARSEMimg.ReadImg(imgfile)        

        imgH = int(img.shape[0])
        imgW = int(img.shape[1])
        #print(basename(imgfile))
        # h = int(basename(imgfile)[0:-4].split('_')[1])
        # w = int(basename(imgfile)[0:-4].split('_')[2])
        h = int(float(basename(imgfile)[0:-4].split('_')[2]))
        w = int(float(basename(imgfile)[0:-4].split('_')[3]))

        if h != int(h) : 
            h += 0.25
            img = img[ int(imgH / 4) : int(imgH * 3 / 4), : ]
        if w != int(w) : 
            w += 0.25
            img = img[ : , int(imgW / 4) : int(imgW * 3 / 4)]
            
        outImg = ARSEMimg.SetImgValueAll(img, outImg, imgH, imgW, h, w)

    SavePNG(out_merge + basename(filename)[0:-4] + '_Predict.png', outImg[0])

    return outImg

# Remember to modify the net structurt and the activate function before evaluating

def precision_idx(CF_Matrix, dim):
    #TP, TN, FP, FN
    if dim == 4:
        accu = (CF_Matrix[0]+CF_Matrix[1])/CF_Matrix.sum()
        prec_g = CF_Matrix[0]/(CF_Matrix[0]+CF_Matrix[2])
        #prec_ng = CF_Matrix[3]/(CF_Matrix[1]+CF_Matrix[3])
        Rec_g = CF_Matrix[0]/(CF_Matrix[0]+CF_Matrix[3])
        #Rec_ng = CF_Matrix[3]/(CF_Matrix[0]+CF_Matrix[3])
        F1_g = str((2 * prec_g * Rec_g) / (prec_g + Rec_g))
        #F1_ng = str((2 * prec_ng * Rec_ng) / (prec_ng + Rec_ng))
        kap_g = str((accu - prec_g)/(1 - prec_g))
        #kap_ng = str((accu - prec_ng)/(1-prec_ng))
    
        precision_idx = ("Accuracy = " + str(accu) + "\n" 
                         "Precision_Ground = " + str(prec_g) + "\n" + 
                         #"Precision_NonGround = " + str(prec_ng) + "\n" 
                         "Recall_Ground = " + str(Rec_g) + "\n" + 
                         #"Recall_NonGround = " + str(Rec_ng) + "\n"
                         "F1_Ground = " + str(F1_g) + "\n" + 
                         #"F1_NonGround = " + str(F1_ng) + "\n"
                         "Kappa_Ground = " + str(kap_g) + "\n"
                         #"Kappa_NonGround = " + str(kap_ng) + "\n"
                         )
    
    else : print("No this Dimension")

    return precision_idx

def BackProjection(filename, outImg, ignore_Zero = False):
    
    print("Reading Point Cloud " + basename(filename))

    laz = read(filename)
    if type(outImg) == type("123"):
        classimg = np.array(ARSEMimg.ReadImg(outImg))
        #classimg = np.array(classimg)[0]
    else:
        classimg = outImg

    print("Successfully Read The Point Cloud")   
    Ep, Np, Hp, classification = ARSEMLaz.GetLAZInfo_noInt(laz) 
    imgW, imgH ,xyMaxMin = ARSEMLaz.GetImgProjectionInfo(Ep, Np, Hp) #[Emax, Nmax, Hmax, Emin, Nmin, Hmin]

    print("Start Caculating ")    

    minImg = ARSEMimg.CreateMinImgTW(Ep, Np, Hp, xyMaxMin[3], xyMaxMin[4], ARSEMimg.CreateImgArray(imgH, imgW, 1, np.float32) + 9999.0)

    print("Classifying")
    classification, CF_Matrix, notice = ARSEMLaz.ClassifyPointsTW(Ep, Np, Hp, minImg, classimg, classification, xyMaxMin[3], xyMaxMin[4], thres, sigma)
    precision_index = precision_idx(CF_Matrix, 4)                 #classimg[0]

    if True:
        f = open(motherfolder + "ConfusionMatrix.txt", 'w')
        f.write(notice)
        f.writelines(precision_index)
        f.writelines(str(CF_Matrix))
        f.close()

    print("Start Writing")
    ARSEMLaz.WriteLAZ(laz, Ep, Np, Hp, classification, motherfolder + os.path.basename(filename)[0:-4] + "_Backproj_0616.las")

def ToBinaryImg(probImg, thres = 50):
    
    binImg = probImg

    binImg[binImg < thres] = 0
    binImg[binImg >= thres] = 2

    return binImg

def MissingValueCover(filename, binImg, inpt_UnClippedtrain):

    LabelImg = ARSEMimg.ReadImg(inpt_UnClippedLabel + basename(filename)[0:-4] + ".png")
    # values, counts = np.unique(featImg, return_counts = True)
    # idx = np.argmax(counts)
        
    cover_binImg = binImg[0]
    cover_binImg[LabelImg == 1] = 1

    return cover_binImg

@njit(nogil = True)
def checkclassification(Labelimg, predimg):
    
    gg, gng, ngg, ngng,  mspxl = 0, 0, 0, 0, 0.0001
    
    for row in range(Labelimg.shape[0]):
        for col in range(Labelimg.shape[1]):

            gtlabel = Labelimg[row, col]
            prlabel = predimg[row, col]

            if gtlabel == 1: mspxl +=1
            elif gtlabel == 2 and prlabel == 2 : gg += 1
            elif gtlabel == 2 and prlabel == 0 : gng += 1
            elif gtlabel == 0 and prlabel == 2 : ngg += 1
            elif gtlabel == 0 and prlabel == 0 : ngng += 1

    g_ngpxl, allpxl = gg + gng + ngg + ngng , Labelimg.shape[0] * Labelimg.shape[1]
    
    Accuracy = round((gg + ngng) / g_ngpxl, 3)
    Accurcay_MS = round((gg + ngng + mspxl) / allpxl, 3)
    Recall = round(gg / (gg + gng), 3)
    Recall_MS = round((gg + mspxl) / (gg + gng + mspxl), 3)
    Precision = round(gg / (gg + ngg), 3)
    Precision_MS = round((gg + mspxl) / (gg + mspxl + ngg), 3)
    f1 = round(2 * Precision * Recall / (Precision + Recall), 3)
    f1_MS = round(2 * Precision_MS * Recall_MS / (Precision_MS + Recall_MS), 3)

    cfm = np.array([[gg, gng],[ngg, ngng]])

    return Accuracy, Recall, Precision, f1, Accurcay_MS, Recall_MS, Precision_MS, f1_MS, cfm

def Eval(pred_Bin_MS_Img, filename, inpt_UnClippedLabel):
    
    Labelimg = ARSEMimg.ReadImg(inpt_UnClippedLabel + basename(filename)[0:-4] + ".png") #[0:8] + ".png")#
    Accuracy, Recall, Precision, f1, Accuracy_MS, Recall_MS, Precision_MS, f1_MS, cfm = checkclassification(Labelimg, pred_Bin_MS_Img)
    
    print('Confusion Matrix')
    print('gg : ' + str(cfm[0, 0]) + ' ngg : ' + str(cfm[1, 0]))
    print('gng : ' + str(cfm[0, 1]) + ' ngng : ' + str(cfm[1, 1]))
    print(' ')
    print('Without Considering Missing Pixels : ')
    print('Accuracy : ' + str(Accuracy) + ' Precision : ' + str(Precision))
    print('Recall : ' + str(Recall) + ' F1 : ' + str(f1))
    print(' ')
    print('With Considering Missing Pixels : ')
    print('Accuracy : ' + str(Accuracy_MS) + '  Precision : ' + str(Precision_MS))
    print('Recall : ' + str(Recall_MS) + '  F1 : ' + str(f1_MS) + '\n')

if __name__ == "__main__":
    
    #Prediction
    #commander_binary(Predict_bin = True, Eval_All = False)

    fileList = glob.glob("E:\project_data/apply\GroundObject\Testing" + "/*.las") # Testing
    #fileList = glob.glob("E:\project_data/apply\GroundObject\Forest" + "/*.las") # Forest
    #fileList = glob.glob("E:\project_data/apply\GroundObject\LowVeg" + "/*.las") # LowVeg
    #fileList = glob.glob("E:\project_data/apply\GroundObject\City" + "/*.las") # City

    #fileList = glob.glob("E:\project_data/apply\Plane\DEMfilt" + "/*.las")  #Plane
    #fileList = glob.glob("E:\project_data/apply\City\DEMfilt" + "/*.las") # Pure #City
    #fileList = glob.glob("E:\project_data/apply\Mont\DEMfilt" + "/*.las") # Mountain
    #fileList = glob.glob("E:\project_data/apply\Hill\DEMfilt" + "/*.las") # Hill

    thres = 2
    sigma = 0.05
    threshold = 73

    for filename in fileList:
        print(basename(filename)[0:8])

        if basename(filename)[0:8] == '94191036' : continue #Not Readable
        featImg = ARSEMimg.ReadImg(inpt_UnClippedtrain + basename(filename)[0:-4] + '.tif')
        
        locationlist, croppedList, imgInfo = ARSEMimg.featImgCropper(featImg, 256, 256, 128)    
        pr = predict_from_featImg_binary(inps= croppedList, checkpoints_path=checkpoint)
        probImg = ARSEMimg.LabelImgMerger(pr, locationlist, imgInfo)        

        #if basename(filename)[0:8] not in ['94181056', '94191039', '94193025', '95203096'] : continue # Forest
        #if basename(filename)[0:8] not in ['94181012', '94184019', '94202029', '95213016'] : continue # City
        #if basename(filename)[0:8] not in ['94181053', '94192032', '95174002', '95204062'] : continue # LowVeg
        #if basename(filename)[0:8] not in ['95174021', '94191048', '95212098'] : continue # Testing
        #if basename(filename)[0:8] not in ['95203043', '94181056'] : continue # Hill Evaluation
        #if basename(filename)[0:8] not in ['95183019', '95212001'] : continue # Mountain Evaluation
        #if basename(filename)[0:8] not in ['94182065', '94182039'] : continue # City Evaluation
        #if basename(filename)[0:8] not in ['94181061', '95203004'] : continue #['94184019'] : continue # Plane Evaluation

        binImg = ToBinaryImg(probImg, threshold)
        outImg = MissingValueCover(filename, binImg, inpt_UnClippedtrain) #binImg#
        SavePNG(out_merge + basename(filename)[0:-4] + '_bin_MS' + str(threshold) + '_Mid.png', outImg) #[0]
        print('\nGround Threshold : ' + str(threshold))
        Eval(outImg, filename, inpt_UnClippedLabel) #[0]

        #BackProjection(filename, outImg)