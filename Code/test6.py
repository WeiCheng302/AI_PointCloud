import ARSEMimgTW

prImg = "D:\Point2IMG\Taiwan/0523_Plane_Unprepro/94181061_DEMfilt_bin_MS73.png"
#prImg = "D:\Point2IMG\Taiwan/0523_Plane_Unprepro/95203004_DEMfilt_bin_MS73.png"
#gtImg = "D:\Point2IMG\Taiwan/0602_Meanfill\Val_Label/95203004_DEMfilt.png"
gtImg = "D:\Point2IMG\Taiwan/0602_Meanfill\Val_Label/94181061_DEMfilt.png"

pr = ARSEMimgTW.ReadImg(prImg)
gt = ARSEMimgTW.ReadImg(gtImg)

imgH = pr.shape[0]
imgW = pr.shape[1]

ms = 0
gg = 0
gng = 0
ngg = 0
ngng = 0

for i in range(imgH):
    for j in range(imgW):
        if gt[i, j] == 1:
            ms += 1
            continue

        if gt[i, j] == 2 and pr[i, j] == 2 : gg += 1
        elif gt[i, j] == 2 and pr[i, j] == 0 : gng += 1
        elif gt[i, j] == 0 and pr[i, j] == 2 : ngg += 1
        elif gt[i, j] == 0 and pr[i, j] == 0 : ngng += 1

accuracy = gg / (gg + gng+ ngg + ngng)
precision = gg / (gg + ngg)
recall = gg / (gg + gng)
f1  = 2 * precision * recall / (precision + recall)

print(gg, gng, ngg, ngng)
print('Accuracy = ' + str(accuracy))
print('Precision = ' + str(precision))
print('Recall = ' + str(recall))
print('F1 = ' + str(f1))