from keras.models import *
from keras.layers import *
import keras.backend as K
import keras

from .config import IMAGE_ORDERING

def _Same_Size_model(input_height=224, input_width=224, channels=3):

    assert (K.image_data_format() ==
            'channels_last'), "Currently only channels last mode is supported"
    assert (IMAGE_ORDERING ==
            'channels_last'), "Currently only channels last mode is supported"
    assert input_height % 32 == 0 
    assert input_width % 32 == 0 

    img_input = Input(shape=(input_height, input_width, channels))
    x = ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING)(img_input)

def _Same_Size_CNN(input_height=224, input_width=224, channels=3):
    model = Sequential()
    model.add(Conv2D())
