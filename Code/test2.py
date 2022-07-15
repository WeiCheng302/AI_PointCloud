import numpy as np
import imageio
from requests import options
import ARSEMimgTW as ARSEMimg
import shapefile
import shapely.geometry as geometry
import osgeo.gdal as gdal
import osgeo.ogr as ogr
import osgeo.osr as osr
import subprocess

InputVector = "D:\Point2IMG\Taiwan\Taiwan_Point_Cloud/103_plane\pointcloud\RiverSHP/96221008w.shp"
OutputImage = "D:\Point2IMG\Taiwan\Taiwan_Point_Cloud/103_plane\pointcloud\96221008w.tif"

# Define pixel_size and NoData value of new raster
pixel_size = 1
NoData_value = 0

# Open the data source and read in the extent
source_ds = ogr.Open(InputVector)
source_layer = source_ds.GetLayer()
source_srs = source_layer.GetSpatialRef()
x_min, x_max, y_min, y_max = source_layer.GetExtent()

# Create the destination data source
x_res = int((x_max - x_min) / pixel_size)
y_res = int((y_max - y_min) / pixel_size)
target_ds = gdal.GetDriverByName('MEM').Create('', x_res, y_res, gdal.GDT_Byte)
target_ds.SetGeoTransform((x_min, -pixel_size, 0, y_max, 0, pixel_size))
band = target_ds.GetRasterBand(1)
band.SetNoDataValue(NoData_value)

# Rasterize
gdal.RasterizeLayer(target_ds, [1], source_layer, burn_values=[1])

# Read as array
array = band.ReadAsArray()
print(array)


# #RefImage = 'Image_Name.tif'

# gdalformat = 'GTiff'
# datatype = gdal.GDT_Byte
# burnVal = 1 #value for the output image pixels
# ##########################################################
# # Get projection info from reference image

# # Open Shapefile
# Shapefile = ogr.Open(InputVector)
# Shapefile_layer = Shapefile.GetLayer()
# extent = Shapefile_layer.GetExtent()
# transform = Shapefile.GetGeoTransform()

# imgH = int(extent[3] - extent[2])
# imgW = int(extent[1] - extent[0])
# # Rasterise
# print("Rasterising shapefile...")

# #Image = gdal.Open(RefImage, gdal.GA_ReadOnly)

# Output = gdal.GetDriverByName(gdalformat).Create(OutputImage, imgW, imgH, 1, datatype, options=['COMPRESS=DEFLATE'])
# #Output.SetProjection(Image.GetProjectionRef())
# #Output.SetGeoTransform(Image.GetGeoTransform()) 

# # Write data to band 1
# Band = Output.GetRasterBand(1)
# Band.SetNoDataValue(100)
# gdal.RasterizeLayer(Output, [1], Shapefile_layer, burn_values=[burnVal])

# # Close datasets
# Band = None
# Output = None
# Image = None
# Shapefile = None

# # Build image overviews
# subprocess.call("gdaladdo --config COMPRESS_OVERVIEW DEFLATE "+OutputImage+" 2 4 8 16 32 64", shell=True)
# print("Done.")


# shp = ogr.Open(RouteSHP)
# shp= shp.GetLayer()
# print(123)
# x_min, x_max, y_min, y_max = shp.GetExtent()
# print(123)
# pixel_size = 1
# x_res = int((x_max - x_min) / pixel_size)
# y_res = int((y_max - y_min) / pixel_size)

# #get GeoTiff driver by 
# driver = gdal.GetDriverByName('GTiff')

# #passing the filename, x and y direction resolution, no. of bands, new raster.
# new_raster = driver.Create(RouteRas, x_res, y_res, 1, gdal.GDT_Int16)

# #transforms between pixel raster space to projection coordinate space.
# #new_raster.SetGeoTransform((x_min, pixel_size, 0, y_min, 0, pixel_size))

# #get required raster band.
# band = new_raster.GetRasterBand(1)

# #assign no data value to empty cells.
# band.SetNoDataValue(0)
# band.FlushCache()

# #main conversion method
# output_raster = gdal.RasterizeLayer(new_raster, [1], shp, burn_values=[255])
# print(000)
# #adding a spatial reference
# output_raster = gdal.Open(output_raster)
# print(111)

# ARSEMimg.SaveTiff_1inp(RouteRas, output_raster, 0)
# print(222)
# print(output_raster)


#gdal.RasterizeLayer(raster, [1], lyr, options=['ATTRIBUTE=chn_id'])
#chn_ras_ds.GetRasterBand(1).SetNoDataValue(0.0)
#chn_ras_ds = None

# river = shapefile.Reader(Route)
# shape = river.shapes()[0]

# test_point_in = (191786.881, 2487523.488)
# test_point_out = (192082.377, 2487419.639)

# imgW = int(shape.bbox[2] - shape.bbox[0])
# imgH = int(shape.bbox[3] - shape.bbox[1])

# out = np.zeros((imgH, imgW)).astype(np.uint8)
# print(out.shape)
# count = 0
# for i in range(500, 550):#int(imgH/2)):
#     for j in range(1, 200):#int(imgW/2)):
#         if geometry.Point( (i + shape.bbox[1], j + shape.bbox[0] ) ).within(geometry.shape(shape)):
#             out[i, j] += count
#             count += 1

# print(out)
# import cv2
# cv2.imshow("123", out)
# cv2.waitKey(0)
# cv2.imwrite("OUT.png", out)
#ARSEMimg.SavePNG("out336.png", (np.zeros((1000,1000)) + 100).astype(np.uint8) )

#if geometry.Point(test_point_in).within(geometry.shape(shape[0])): print("test_in InNNN")
#if geometry.Point(test_point_out).within(geometry.shape(shape[0])): print("test_out InNNN")

#print(shape[0].bbox)
#print(river.records())