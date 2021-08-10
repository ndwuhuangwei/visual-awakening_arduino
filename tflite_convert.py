import tensorflow as tf
import os
import datetime
from Involution_MobileNetV2 import MobileNetV2_Inv
from micronet_model import MicroNet

rho = 224

ckpt_name = 'Model_MicroNet_VWW_20210802-133646'
keras_model = os.path.join(os.getcwd(), 'weights', ckpt_name, ckpt_name + '.ckpt')

# model = tf.keras.applications.mobilenet_v2.MobileNetV2(alpha=1.0, weights=None, classes=2,
#                                                        classifier_activation='sigmoid')
# model = MobileNetV2_Inv(include_top=True, weights=None, alpha=1,
#                         input_shape=(rho, rho, 3), classes=2,
#                         classifier_activation='sigmoid', dropout=0.5)

model = MicroNet(input_shape=(rho, rho, 3), alpha=1.0, weights=None, classes=2, classifier_activation='sigmoid')

model.load_weights(keras_model)

# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
#
# quantized_tflite_model = converter.convert()

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()

if not os.path.exists('./tflite_models'):
    os.mkdir('./tflite_models')

tflite_model_name = "tfliteModel_{}_{}".format(ckpt_name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tflite_model_path = os.path.join('./tflite_models', tflite_model_name)
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_quant_model)
