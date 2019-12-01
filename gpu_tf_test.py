"""
Определение доступных GPU и принудительный запуск на заданом типе устройства GPU/CPU
"""

import tensorflow as tf
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

print("GPU Available: ", tf.test.is_gpu_available())

#принудительно для CPU
#with tf.device('/device:CPU:0'):

if tf.test.is_gpu_available():
    with tf.device('/device:GPU:0'):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
        c = tf.matmul(a, b)
    # Creates a session with log_device_placement set to True.
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # Runs the op.
    print(sess.run(c))
