








from tensorflow.keras.backend import binary_crossentropy
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
import numpy as np

y_true = [0.5, 1, 0, 0.5]
y_pred = [0.6, 0.4, 0.4, 0.6]

y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
y_true = math_ops.cast(y_true, y_pred.dtype)

print("BCE = ")
print(binary_crossentropy(y_true, y_pred).numpy())
print("  ")
# print("BCE * MASK = ")
# print(binary_crossentropy(y_true, y_pred).numpy() * np.array(y_true))


def Masked_Binary_Cross_Entropy(gt, pr):
    from tensorflow.keras.backend import binary_crossentropy
    from tensorflow.python.framework import ops
    from tensorflow.python.ops import math_ops
    
    pr = ops.convert_to_tensor_v2_with_dispatch(pr)
    gt = math_ops.cast(gt, pr.dtype)

    def gtmask(arr):
        arr[arr != 0.5] = 1
        arr[arr == 0.5] = 0
        return arr

    mask = gtmask(gt.numpy())
    print(mask)
    print(binary_crossentropy(gt, pr).numpy())

    return ( (binary_crossentropy(gt, pr).numpy() * mask).sum() )/ len(mask[mask == 1])

y_true = [0.5, 1, 0, 0.5]
y_pred = [0.6, 0.4, 0.4, 0.6]

print(Masked_Binary_Cross_Entropy(y_true, y_pred))

import tensorflow as tf
a= tf.zeros(5)
a[0,] = 5
print(a)
print(a[a!=0])