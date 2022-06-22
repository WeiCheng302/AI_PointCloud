from keras.callbacks import Callback

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')

        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')

        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()
        plt.savefig("Loss_0325.png")

class CheckpointsCallback(Callback):
    def __init__(self, checkpoints_path):
        self.checkpoints_path = checkpoints_path

    def on_epoch_end(self, epoch, logs=None):
        if self.checkpoints_path is not None:
            self.model.save_weights(self.checkpoints_path + "." + str(epoch))
            print("saved ", self.checkpoints_path + "." + str(epoch))

import json
import os

from keras_segmentation.data_utils.data_loader import image_segmentation_generator, random_img_seg_generator_pre_read, random_img_seg_generator_pre_read_multi, \
    verify_segmentation_dataset, random_img_seg_generator, read_all_imgs
from tensorflow.keras.backend import binary_crossentropy
from tensorflow.python import framework, ops
from keras_segmentation.models.all_models import model_from_name
import six
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import keras.backend as backend
backend.clear_session()

#GPU Setting
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

import matplotlib.pyplot as plt
import glob
import wandb
from wandb.keras import WandbCallback
#wandb.init(project="0325_4000", entity = 'aipointcloud')
history = LossHistory()

# TT: MBCE, TF: BCE, FT:MCCE, FF:CCE
BinaryCrossEntropy = True
ignore_zero_class = True

model = 'transunet'

checkpoints_path = "D:/image-segmentation-keras/0602_unet_pure/CheckPoints/_0602_TW_Plane_3lyr_unet_meanfill_meanfill_demPrepro_MBCE_RD_Adam"
Model_Path = "D:/image-segmentation-keras/0602_unet_pure/Model/_0602_TW_Plane_3lyr_unet_meanfill_meanfill_demPrepro_MBCE_RD_Adam.h5"
motherfolder = "D:\Point2IMG\Taiwan/0609_Test/"

input_size = (256, 256, 3)
filter_num = [64, 128, 256]
n_classes = 1
#==== Default Parameter ====#
stack_num_down=2
stack_num_up=2
activation='ReLU'
embed_dim = 128
num_mlp = 128
num_heads = 3
num_transformer = 3
mlp_activation = 'GELU'
output_activation = 'Softmax'
batch_norm=False
pool=True
unpool=True
backbone=None
weights='imagenet'
freeze_backbone=True
freeze_batch_norm=True
name='transunet'
#==========================#

train_images = motherfolder + "UnClipped_Img"
train_annotations = motherfolder + "UnClipped_Png"

epochs = 400
batch_size = 1

validate = True
val_images = motherfolder + "Val_Img"
val_annotations = motherfolder + "Val_Label"
val_batch_size = batch_size


input_height = input_size[0]
input_width = input_size[1]
# n_classes = 1
channel_count = input_size[2]

weightArr = ([1, 1, 1])
do_augment= True
augmentation_name="aug_flip_and_rot"

verify_dataset = False

default_callback = ModelCheckpoint(filepath=checkpoints_path + ".{epoch:05d}",
            save_weights_only=True, period = 80, verbose=True)

callbacks = [default_callback, history]#, WandbCallback()]
#tf.keras.callbacks.TensorBoard(log_dir='.\\0331/logs'),
auto_resume_checkpoint = False #True #Auto Resume the checkpoint from checkpoints_path

####### Never Change########
metrics_idx = 'accuracy' # 'Recall' # 'Precision'
load_weights= None 
steps_per_epoch=512
val_steps_per_epoch=512
gen_use_multiprocessing=False
optimizer_name= 'adam' #optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=False)#'adam'
custom_augmentation=None
other_inputs_paths=None
preprocessing=None
read_image_type=1  # cv2.IMREAD_COLOR = 1 (rgb),
                   # cv2.IMREAD_GRAYSCALE = 0,
                   # cv2.IMREAD_UNCHANGED = -1 (4 channels like RGBA)
############################

def find_latest_checkpoint(checkpoints_path, fail_safe=True):

    # This is legacy code, there should always be a "checkpoint" file in your directory
    def get_epoch_number_from_path(path):
        return path.replace(checkpoints_path, "").strip(".")

    # Get all matching files
    all_checkpoint_files = glob.glob(checkpoints_path + ".*")
    if len(all_checkpoint_files) == 0:
        all_checkpoint_files = glob.glob(checkpoints_path + "*.*")
    all_checkpoint_files = [ff.replace(".index", "") for ff in
                            all_checkpoint_files]  # to make it work for newer versions of keras
    # Filter out entries where the epoc_number part is pure number
    all_checkpoint_files = list(filter(lambda f: get_epoch_number_from_path(f)
                                       .isdigit(), all_checkpoint_files))
    if not len(all_checkpoint_files):
        # The glob list is empty, don't have a checkpoints_path
        if not fail_safe:
            raise ValueError("Checkpoint path {0} invalid"
                             .format(checkpoints_path))
        else:
            return None

    # Find the checkpoint file with the maximum epoch
    latest_epoch_checkpoint = max(all_checkpoint_files,
                                  key=lambda f:
                                  int(get_epoch_number_from_path(f)))

    return latest_epoch_checkpoint

def weighted_categorical_crossentropy(weights):
    from keras import backend as K
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss

def masked_categorical_crossentropy(gt, pr):
    from keras.losses import categorical_crossentropy    
    mask = 1 - gt[:, :, 1]  #mask = 1 - gt[:, :, 0]

    #If Label == 0 : Filtered, not taking part into loss computation
    #print(categorical_crossentropy(gt, pr) * mask)
    return categorical_crossentropy(gt, pr) * mask

def Binary_Cross_Entropy(gt, pr):
    from keras.losses import binary_crossentropy
    return binary_crossentropy(gt, pr)

def Masked_Binary_Cross_Entropy(gt, pr): 
    
    # gt[gt != 0.5] = 1 else 0

    mask = tf.cast( tf.where( gt != 0.5, 1, 0), tf.float32 )
    loss = tf.reduce_sum(binary_crossentropy(gt, pr) * mask) / (float(len(tf.where(mask == 1))) + 0.00000000001)
    
    return loss

model = model_from_name[model](
                 input_size, filter_num, n_classes, stack_num_down, stack_num_up, embed_dim, num_mlp,
                 num_heads, num_transformer, activation, mlp_activation, output_activation,
                 batch_norm, pool, unpool, backbone, weights, freeze_backbone, freeze_batch_norm, name)
print(model)
input_height = model.input_shape[1]
input_width = model.input_shape[2]
output_height = input_height
output_width = input_width
model.summary()

if validate:
    assert val_images is not None
    assert val_annotations is not None

if optimizer_name is not None:

    if ignore_zero_class and Binary_Cross_Entropy:
        loss_k = Masked_Binary_Cross_Entropy
        print("Loss = Masked Binary Cross Entropy")
    
    elif ignore_zero_class and not BinaryCrossEntropy:
        loss_k = masked_categorical_crossentropy
        print("Loss = Masked Categorical Cross Entropy")

    elif BinaryCrossEntropy and not ignore_zero_class: 
        loss_k = Binary_Cross_Entropy
        print("Loss = Binary Cross Entropy")
    
    else:
        loss_k = weighted_categorical_crossentropy(weightArr) #'categorical_crossentropy'
        print("Loss = Weighted Categorical Cross Entropy")

    print(optimizer_name)
    model.compile(loss=loss_k, optimizer=optimizer_name, metrics=[metrics_idx])

if checkpoints_path is not None:
    config_file = checkpoints_path + "_config.json"
    dir_name = os.path.dirname(config_file)

    if ( not os.path.exists(dir_name) ) and len( dir_name ) > 0 :
        os.makedirs(dir_name)

    with open(config_file, "w") as f:
        json.dump({
            "model_class": name,
            "n_classes": n_classes,
            "input_height": input_height,
            "input_width": input_width,
            "output_height": output_height,
            "output_width": output_width
        }, f)

if load_weights is not None and len(load_weights) > 0:
    print("Loading weights from ", load_weights)
    model.load_weights(load_weights)

initial_epoch = 0

if auto_resume_checkpoint and (checkpoints_path is not None):
    latest_checkpoint = find_latest_checkpoint(checkpoints_path)
    if latest_checkpoint is not None:
        print("Loading the weights from latest checkpoint ", latest_checkpoint)
        model.load_weights(latest_checkpoint)

        initial_epoch = int(latest_checkpoint.split('.')[-1])

if verify_dataset:
    print("Verifying training dataset")
    verified = verify_segmentation_dataset(train_images, train_annotations, n_classes)
    assert verified

    if validate:
        print("Verifying validation dataset")
        verified = verify_segmentation_dataset(val_images, val_annotations, n_classes)
        assert verified

train_img_list, train_seg_list = read_all_imgs(train_images, train_annotations)

###train_gen = random_img_seg_generator(
###train_gen = random_img_seg_generator_pre_read_multi(

train_gen = random_img_seg_generator_pre_read(
            train_img_list, train_seg_list, batch_size,
            n_classes, input_height, input_width, output_height, output_width,
            do_augment = do_augment, augmentation_name="aug_flip_and_rot")

'''
train_gen = image_segmentation_generator(
    train_images, train_annotations,  batch_size,  n_classes,
    input_height, input_width, output_height, output_width,
    do_augment=do_augment, augmentation_name=augmentation_name,
    custom_augmentation=custom_augmentation, other_inputs_paths=other_inputs_paths,
    preprocessing=preprocessing, read_image_type=read_image_type)'''

if validate:
    val_img_list, val_seg_list = read_all_imgs(val_images, val_annotations)

    ###val_gen = random_img_seg_generator(
    ###val_gen = random_img_seg_generator_pre_read_multi(
    
    val_gen = random_img_seg_generator_pre_read(
            val_img_list, val_seg_list, batch_size,
            n_classes, input_height, input_width, output_height, output_width,
            do_augment = do_augment, augmentation_name="aug_flip_and_rot")
'''
    val_gen = image_segmentation_generator(
        val_images, val_annotations,  val_batch_size,
        n_classes, input_height, input_width, output_height, output_width,
        other_inputs_paths=other_inputs_paths,
        preprocessing=preprocessing, read_image_type=read_image_type)'''

if callbacks is None : callbacks = []

if not validate:
    history = model.fit(train_gen, steps_per_epoch=steps_per_epoch,
              epochs=epochs, callbacks=callbacks, initial_epoch=initial_epoch)
    model.save(Model_Path)
else:
    history = model.fit(train_gen,
              steps_per_epoch=steps_per_epoch,
              validation_data=val_gen,
              validation_steps=val_steps_per_epoch,
              epochs=epochs, callbacks=callbacks, use_multiprocessing=gen_use_multiprocessing, initial_epoch=initial_epoch)
    model.save(Model_Path)

#history.loss_plot('epoch')

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.savefig('Result.png')
plt.show()