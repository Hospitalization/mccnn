import tensorflow as tf
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

tf.app.flags.DEFINE_integer(
    'input_shape',
    '336',
    'input shape.')

tf.app.flags.DEFINE_integer(
    'epochs',
    '100',
    'epochs')

tf.app.flags.DEFINE_float(
    'lr',
    '1e-5',
    'learning rate.')

tf.app.flags.DEFINE_float(
    'beta',
    '0.3',
    'beta to get sigma.')

tf.app.flags.DEFINE_string(
    'model',
    'mscnn',
    'Model type')

FLAGS = tf.app.flags.FLAGS
