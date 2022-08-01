import ARSEMLazTW

las = "D:\Point2IMG\Taiwan/0706_All/95183053_DEMfilt_Backproj.las"

las = ARSEMLazTW.ReadLAZ(las)

xp, yp, zp, intensity, classification = ARSEMLazTW.GetLAZInfo(las)

a = 1