from osgeo import gdal, ogr
import ARSEMimgTW as ARSEMimg

shp_name = 'E:\project_data\WaterLine/105_110/94192010w.shp'
ras_name = 'E:\project_data\WaterLine/105_110/__0_out.tif'

shp = ogr.Open(shp_name)
lyr = shp.GetLayer()
extent = lyr.GetExtent()
print(extent)

driver = gdal.GetDriverByName('GTiff')
ds = driver.Create(ras_name, 1, 1, 1, gdal.GDT_UInt16)
ds.SetGeoTransform( (int(extent[0]), 1, 0, int(extent[2]), 0, -1) )
gdal.RasterizeLayer(ras_name, [1], lyr, burn_values=[5])

