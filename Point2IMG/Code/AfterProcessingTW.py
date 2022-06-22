import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     # Currently, memory growth needs to be the same across GPUs
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
#     print(e)
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

from time import time
import numpy as np
import glob, os, laspy
import ARSEMimgTW as ARSEMimg
import ARSEMLazTW as ARSEMLaz
from laspy import read
from numba import njit
from os.path import basename
from ARSEMimgTW import ReadImg, CreateImgArray, SetImgValueAll, SavePNG

motherfolder = "D:\Point2IMG\Taiwan/0523_Plane/"

checkpoint = "D:/image-segmentation-keras/0523_Plane/CheckPoints\_0523_TW_Plane_MbSeg_demPrepro_MBCE_RD_Adam"

inpt_allimg = motherfolder + "All_Img"
inpt_allLabel = motherfolder + "All_Label"

inpt_trainimg = motherfolder + "Train_Img"
inpt_trainLabel = motherfolder + "Train_Label"

inpt_UnClippedLabel = motherfolder + "UnClipped_Png_All/"
inpt_UnClippedtrain = motherfolder + "UnClipped_Img_All/"

inpt_testimg = motherfolder + "Test_Img"
inpt_testLabel = motherfolder + "Test_Label"

out_merge = motherfolder
out_predict_folder = motherfolder + "Predicted_Label"

thres = 2
sigma = 0.05

def digit(a):
    return str("{:.3f}".format(a))

def commander_binary(Predict_bin = False, Eval_All = False, Eval_Train = False):
    DirCurrent = os.getcwd()
    os.chdir("D:\image-segmentation-keras")
    
    #Prediction
    if Predict_bin :
        os.system("python -m keras_segmentation predict_binary --checkpoints_path=" + checkpoint + " --input_path=" + inpt_allimg + " --output_path=" + out_predict_folder)
    
    #Evaluation All
    if Eval_All :
        os.system("python -m keras_segmentation evaluate_model_binary --checkpoints_path=" + checkpoint +" --images_path=" + inpt_allimg + " --segs_path=" + inpt_allLabel)

    #Evaluation Train
    if Eval_Train :
        os.system("python -m keras_segmentation evaluate_model_binary --checkpoints_path=" + checkpoint +" --images_path=" + inpt_trainimg + " --segs_path=" + inpt_trainLabel)

    os.chdir(DirCurrent)
    print("Commander Finished")

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
        h = int(basename(imgfile)[0:-4].split('_')[2])
        w = int(basename(imgfile)[0:-4].split('_')[3])
            
        if ignore_0:
            mask = CreateMask(imgfile)
            outImg = ARSEMimg.SetImgValueAll(mask[0] * img, outImg, imgH, imgW, h, w)
            #print(imgfile)
        else :
            outImg = ARSEMimg.SetImgValueAll(img, outImg, imgH, imgW, h, w)

    SavePNG(out_merge + basename(filename)[0:-4] + '_Predict.png', outImg[0])

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

    # print("Ground Filtering")
    # las = ARSEMLaz.ReadLAZ(motherfolder + os.path.basename(filename)[0:-4] + "_Backproj.las")
    # new = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
    # new.points = las.points[las.classification == 2]

    # new.write(motherfolder + os.path.basename(filename)[0:-4] + "_Backproj_ground.las")

@njit(nogil = True)
def Confusion_mat(Labelimg, out):

    CF_Mat = np.zeros(4)

    for i in range(Labelimg.shape[0]):
        for j in range(Labelimg.shape[1]):
            if Labelimg[i, j] == 0 and out[i, j] == 0 : CF_Mat[0] += 1 # TP
            elif Labelimg[i, j] == 0 and out[i, j] != 0 : CF_Mat[2] += 1 # FP
            elif Labelimg[i, j] == 2 and out[i, j] != 2 : CF_Mat[3] += 1 # FN
            elif Labelimg[i, j] == 2 and out[i, j] == 2 : CF_Mat[1] += 1 # TN

    return CF_Mat

def findThres(probImg):
    frequency = np.bincount(probImg.flatten())    
    trs = np.where(frequency==np.min(frequency[10:90]))[0]
    return float(np.where(frequency==np.min(frequency[10:90]))[0])

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
    print('gg : ' + str(cfm[0, 0]) + ' gng : ' + str(cfm[0, 1]))
    print('ngg : ' + str(cfm[1, 0]) + ' ngng : ' + str(cfm[1, 1]))
    print(' ')
    print('Without Considering Missing Pixels : ')
    print('Accuracy : ' + str(Accuracy) + ' Precision : ' + str(Precision))
    print('Recall : ' + str(Recall) + ' F1 : ' + str(f1))
    print(' ')
    print('With Considering Missing Pixels : ')
    print('Accuracy : ' + str(Accuracy_MS) + '  Precision : ' + str(Precision_MS))
    print('Recall : ' + str(Recall_MS) + '  F1 : ' + str(f1_MS) + '\n')

def PredAcc(filename, outImg, inpt_UnClippedLabel):
    
    Labelimg = ARSEMimg.ReadImg(inpt_UnClippedLabel + basename(filename)[0:-4] + ".png")
    out = outImg[0]

    #TP, TN, FP, FN
    CF_Mat = Confusion_mat(Labelimg, out)
    Accu = (CF_Mat[0] + CF_Mat[1])/(CF_Mat.sum() - len(Labelimg[Labelimg == 1]))
    Prec = CF_Mat[0]/(CF_Mat[0] + CF_Mat[2])
    Rec = CF_Mat[0]/(CF_Mat[0] + CF_Mat[3])
    F1S = 2 * ((Prec * Rec)/(Prec + Rec))

    print(CF_Mat)
    print(Accu, Prec, Rec, F1S)


if __name__ == "__main__":
    
    #Prediction
    #commander_binary(Predict_bin = True, Eval_All = False)

    fileList = glob.glob("E:\project_data/apply\Plane\DEMfilt" + "/*.las") #/Pure
    #fileList = glob.glob("E:\project_data/apply\City\DEMfilt" + "/*.las")


    thres = 2
    sigma = 0.05
    threshold = 73

    for filename in fileList:
        print(basename(filename)[0:8])
        if basename(filename)[0:8] == '94191036' : continue
        #if basename(filename)[0:8] not in ['94182065', '94182039'] : continue
        if basename(filename)[0:8] not in ['94181061', '95203004'] : continue #['94184019'] : continue
        #if basename(filename)[0:8] != : continue

        #if basename(filename)[0:8] not in  ['94202006']:continue#, '94181061', '95203004', '94202006', '94202005']:continue# , '95203004' , '94184060'] : continue
        probImg = MergeImg(filename, out_predict_folder, out_merge)
        binImg = ToBinaryImg(probImg, threshold)
        outImg = MissingValueCover(filename, binImg, inpt_UnClippedtrain) #binImg#
        SavePNG(out_merge + basename(filename)[0:-4] + '_bin_MS' + str(threshold) + '_Mid.png', outImg) #[0]
        print('\nGround Threshold : ' + str(threshold))
        Eval(outImg, filename, inpt_UnClippedLabel) #[0]

        BackProjection(filename, outImg)
    #Merging Image    
    # for filename in fileList:
    #     if basename(filename)[0:8] not in  ['94181061', '95203004', '94202006', '94202005'] : continue

    #     for i in [18,65,67,69,71,73,75,77,79,81,83,85,87,89,91]:
    #         probImg = MergeImg(filename, out_predict_folder, out_merge)
    #         binImg = ToBinaryImg(probImg, i)  #findThres(probImg))  #SavePNG(out_merge + basename(filename)[0:-4] + '_bin.png', binImg[0])
    #         outImg = MissingValueCover(filename, binImg, inpt_UnClippedtrain)
    #         SavePNG(out_merge + basename(filename)[0:-4] + '_bin_MS' + str(i) + '.png', outImg)
    #         print('\nGround Threshold : ' + str(i))
    #         Eval(outImg, filename, inpt_UnClippedLabel)
        a = 0
    
    fileList = ["E:\project_data/apply\Plane\DEMfilt/" + "94184080_DEMfilt.las"]
    filename = fileList[0]
    BackProjection(filename, outImg)