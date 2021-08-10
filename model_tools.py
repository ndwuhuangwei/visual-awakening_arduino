from tensorflow.python.keras import backend
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.engine import training
# from tensorflow.python.keras.layers import VersionAwareLayers
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, backend, Sequential


class SELayer(layers.Layer):
    """
    SE layer contains Squeeze and excitaton operations
    """
    def __init__(self, channels, ratio=16):
        """
        :param input_tensor: input_tensor.shape=[h,w,c]
        :param ratio:Number of output channels for excitation intermediate operation
        """
        super(SELayer, self).__init__()
        # self.in_tensor = input_tensor
        # self.in_channels = backend.int_shape(input_tensor)[-1]
        self.in_channels = channels
        self.ratio = ratio
      
        self.squeeze = Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Reshape((1, 1, self.in_channels))
        ])
        
        self.excitation = Sequential([
            layers.Conv2D(filters=self.in_channels // self.ratio, kernel_size=(1, 1)),
            layers.Activation("relu"),
            layers.Conv2D(filters=self.in_channels, kernel_size=(1, 1)),
            layers.Activation('sigmoid')
        ])
        
    def call(self, inputs, training=False):
        """
        Use conv by default, PW conv
        :param self:
        :return:
        """
        out = self.squeeze(inputs)
        out = self.excitation(out)
        scale = layers.multiply([inputs, out])
        
        return scale

        

