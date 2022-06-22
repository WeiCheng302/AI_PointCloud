from . import pspnet
from . import unet
from . import segnet
from . import fcn

from keras_unet_collection._model_unet_2d import unet_2d
from keras_unet_collection._model_vnet_2d import vnet_2d
from keras_unet_collection._model_unet_plus_2d import unet_plus_2d
from keras_unet_collection._model_r2_unet_2d import r2_unet_2d
from keras_unet_collection._model_att_unet_2d import att_unet_2d
from keras_unet_collection._model_resunet_a_2d import resunet_a_2d
from keras_unet_collection._model_u2net_2d import u2net_2d
from keras_unet_collection._model_unet_3plus_2d import unet_3plus_2d
from keras_unet_collection._model_transunet_2d import transunet_2d
from keras_unet_collection._model_swin_unet_2d import swin_unet_2d

model_from_name = {}

model_from_name["unet_collect"] = unet_2d
model_from_name["vnet"] = vnet_2d
model_from_name["unetplus"] = unet_plus_2d
model_from_name["r2_unet"] = r2_unet_2d
model_from_name["att_unet"] = att_unet_2d
model_from_name["resunet_a"] = resunet_a_2d
model_from_name["u2net"] = u2net_2d
model_from_name["unet_3plus"] = unet_3plus_2d
model_from_name["transunet"] = transunet_2d
model_from_name["swin_unet"] = swin_unet_2d


model_from_name["fcn_8"] = fcn.fcn_8
model_from_name["fcn_32"] = fcn.fcn_32
model_from_name["fcn_8_vgg"] = fcn.fcn_8_vgg
model_from_name["fcn_32_vgg"] = fcn.fcn_32_vgg
model_from_name["fcn_8_resnet50"] = fcn.fcn_8_resnet50
model_from_name["fcn_32_resnet50"] = fcn.fcn_32_resnet50
model_from_name["fcn_8_mobilenet"] = fcn.fcn_8_mobilenet
model_from_name["fcn_32_mobilenet"] = fcn.fcn_32_mobilenet


model_from_name["pspnet"] = pspnet.pspnet
model_from_name["vgg_pspnet"] = pspnet.vgg_pspnet
model_from_name["resnet50_pspnet"] = pspnet.resnet50_pspnet

model_from_name["vgg_pspnet"] = pspnet.vgg_pspnet
model_from_name["resnet50_pspnet"] = pspnet.resnet50_pspnet

model_from_name["pspnet_50"] = pspnet.pspnet_50
model_from_name["pspnet_101"] = pspnet.pspnet_101


# model_from_name["mobilenet_pspnet"] = pspnet.mobilenet_pspnet


model_from_name["unet_mini"] = unet.unet_mini
model_from_name["unet"] = unet.unet
model_from_name["vgg_unet"] = unet.vgg_unet
model_from_name["resnet50_unet"] = unet.resnet50_unet
model_from_name["mobilenet_unet"] = unet.mobilenet_unet


model_from_name["segnet"] = segnet.segnet
model_from_name["vgg_segnet"] = segnet.vgg_segnet
model_from_name["resnet50_segnet"] = segnet.resnet50_segnet
model_from_name["mobilenet_segnet"] = segnet.mobilenet_segnet
model_from_name["mobilenet_segnet_deep"] = segnet.mobilenet_segnet_deep


