from cv2 import ellipse
import fiona

ellipeceSHP = fiona.open('DEMe95191003.shp')
geoidSHP = fiona.open('DEMg95191003.shp')
print(ellipeceSHP)
print(geoidSHP)