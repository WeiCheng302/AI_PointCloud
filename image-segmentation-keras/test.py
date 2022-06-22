import tensorflow as tf
import numpy as np
from tensorflow.keras import backend

y_true = np.array([[0, 1, 0, 1, 0, 0.5, 0.5, 0.5],
                   [1, 0, 1, 0, 1, 0.5, 0.5, 0.5]], dtype=np.float32)

y_pred = np.array([[0, 0, 1, 0, 0, 0.5, 1, 0],
                   [1, 1, 0, 1, 1, 0.5, 0, 1]], dtype=np.float32)

def binary_focal_loss_fixed(y_true, y_pred, gamma=2., alpha=.25):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        y_true = tf.cast(y_true, tf.float32)
        # Define epsilon so that the back-propagation will not result in NaN for 0 divisor case
        epsilon = backend.epsilon()
        # Add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # Clip the prediciton value
        y_pred = backend.clip(y_pred, epsilon, 1.0 - epsilon)
        # Calculate p_t
        p_t = tf.where(backend.equal(y_true, 1), y_pred, 1 - y_pred)
        # Calculate alpha_t
        alpha_factor = backend.ones_like(y_true) * alpha
        alpha_t = tf.where(backend.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        # Calculate cross entropy
        cross_entropy = -backend.log(p_t)
        weight = alpha_t * backend.pow((1 - p_t), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy

        # Mask the Missing Value
        mask = tf.cast( tf.where( y_true != 0.5, 1, 0), tf.float32 )
        loss = loss * mask
        loss = tf.reduce_sum(loss * mask) / (float(len(tf.where(mask == 1))) + 0.00000000001)
        
        # Sum the losses in mini_batch
        #loss = backend.mean(backend.sum(loss, axis=1))
        return loss


def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """  

    return binary_focal_loss_fixed

print(binary_focal_loss_fixed(y_true, y_pred, gamma=2., alpha=.25))

# from keras_segmentation.predict import predict, predict_multiple, predict_multiple_binary, evaluate_binary
# from numpy import random

# DATA_LOADER_SEED = 0

# random.seed(DATA_LOADER_SEED)
# class_colors = [(random.randint(0, 255), random.randint(
#     0, 255), random.randint(0, 255)) for _ in range(5000)]

# inp = "D:/Point2IMG/Taiwan/Test10_0427/All_Img"
# checkpoints_path = "D:/image-segmentation-keras/_0427/CheckPoints\_0427_Taiwan_BCE_RD"
# predictFolder = "D:/Point2IMG/Taiwan/Test10_0427/Predicted_Label"
# inpall = "D:/Point2IMG/Taiwan/Test10_0427/All_Img"
# segall = "D:/Point2IMG/Taiwan/Test10_0427/All_Label"


# evaluate_binary(model=None, inp_images=None, annotations=None, ignore_0 = False,
#              inp_images_dir=inpall, annotations_dir=segall, checkpoints_path=checkpoints_path, read_image_type=1)

# predict_multiple_binary(model=None, inps=None, inp_dir=inp, out_dir=predictFolder,
#                      checkpoints_path=checkpoints_path, overlay_img=False,
#                      class_names=None, show_legends=False, colors=class_colors,
#                      prediction_width=None, prediction_height=None, read_image_type=1)

# predict(model=None, inp=inp,
#             out_fname=None, checkpoints_path= checkpoints_path, 
#             overlay_img=False, class_names=None, show_legends=False, colors=class_colors,
#             prediction_width=None, prediction_height=None, read_image_type=1)