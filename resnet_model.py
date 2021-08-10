from tensorflow.keras.layers import *
from tensorflow.keras import Input
from tensorflow.keras import Model
import tensorflow as tf
import numpy as np


def conv2d_bn(input, kernel_num, kernel_size=3, strides=1, layer_name='', padding_mode='same'):
    conv1 = Conv2D(kernel_num, kernel_size, strides=strides, padding=padding_mode, name=layer_name + '_conv1')(input)
    batch1 = BatchNormalization(name=layer_name + '_bn1')(conv1)
    return batch1


def shortcut(fx, x, padding_mode='same', layer_name=''):
    layer_name += '_shortcut'
    if x.shape[-1] != fx.shape[-1]:
        k = fx.shape[-1]
        k = int(k)
        identity = conv2d_bn(x, kernel_num=k, kernel_size=1, padding_mode=padding_mode, layer_name=layer_name)
    else:
        identity = x
    return Add(name=layer_name + '_add')([identity, fx])


def bottleneck(input, kernel_num, strides=1, layer_name='bottleneck', padding_mode='same'):
    k1, k2, k3 = kernel_num
    conv1 = conv2d_bn(input, kernel_num=k1, kernel_size=1, strides=strides, padding_mode=padding_mode,
                      layer_name=layer_name + '_1')
    relu1 = ReLU(name=layer_name + '_relu1')(conv1)
    conv2 = conv2d_bn(relu1, kernel_num=k2, kernel_size=3, strides=strides, padding_mode=padding_mode,
                      layer_name=layer_name + '_2')
    relu2 = ReLU(name=layer_name + '_relu2')(conv2)
    conv3 = conv2d_bn(relu2, kernel_num=k3, kernel_size=1, strides=strides, padding_mode=padding_mode,
                      layer_name=layer_name + '_3')
    # print(conv3.shape, input.shape)
    shortcut_add = shortcut(fx=conv3, x=input, layer_name=layer_name)
    relu3 = ReLU(name=layer_name + '_relu3')(shortcut_add)
    
    return relu3


def basic_block(input, kernel_num=64, strides=1, layer_name='basic', padding_mode='same'):
    # k1, k2 = kernel
    conv1 = conv2d_bn(input, kernel_num=kernel_num, strides=strides, kernel_size=3,
                      layer_name=layer_name + '_1', padding_mode=padding_mode)
    relu1 = ReLU(name=layer_name + '_relu1')(conv1)
    conv2 = conv2d_bn(relu1, kernel_num=kernel_num, strides=strides, kernel_size=3,
                      layer_name=layer_name + '_2', padding_mode=padding_mode)
    relu2 = ReLU(name=layer_name + '_relu2')(conv2)
    
    shortcut_add = shortcut(fx=relu2, x=input, layer_name=layer_name)
    relu3 = ReLU(name=layer_name + '_relu3')(shortcut_add)
    return relu3


def make_layer(input, block, block_num, kernel_num, layer_name=''):
    x = input
    for i in range(1, block_num + 1):
        x = block(x, kernel_num=kernel_num, strides=1, layer_name=layer_name + str(i), padding_mode='same')
    return x


def ResNet(input_shape, nclass, activation='sigmoid', net_name='resnet18'):
    """
        :param net_name:
        :param input_shape:
        :param nclass:
        :return:

    Args:
        activation:
    """
    block_setting = {}
    block_setting['resnet18'] = {'block': basic_block, 'block_num': [2, 2, 2, 2], 'kernel_num': [64, 128, 256, 512]}
    block_setting['resnet26'] = {'block': bottleneck, 'block_num': [1, 2, 4, 1],
                                 'kernel_num': [[64, 64, 256], [128, 128, 512],
                                                [256, 256, 1024], [512, 512, 2048]]}
    block_setting['resnet34'] = {'block': basic_block, 'block_num': [3, 4, 6, 3], 'kernel_num': [64, 128, 256, 512]}
    block_setting['resnet50'] = {'block': bottleneck, 'block_num': [3, 4, 6, 3],
                                 'kernel_num': [[64, 64, 256], [128, 128, 512],
                                                [256, 256, 1024], [512, 512, 2048]]}
    block_setting['resnet101'] = {'block': bottleneck, 'block_num': [3, 4, 23, 3],
                                  'kernel_num': [[64, 64, 256], [128, 128, 512],
                                                 [256, 256, 1024], [512, 512, 2048]]}
    block_setting['resnet152'] = {'block': bottleneck, 'block_num': [3, 8, 36, 3],
                                  'kernel_num': [[64, 64, 256], [128, 128, 512],
                                                 [256, 256, 1024], [512, 512, 2048]]}
    net_name = 'resnet18' if not block_setting.__contains__(net_name) else net_name
    block_num = block_setting[net_name]['block_num']
    kernel_num = block_setting[net_name]['kernel_num']
    block = block_setting[net_name]['block']
    
    input_ = Input(shape=input_shape)
    conv1 = conv2d_bn(input_, 64, kernel_size=7, strides=2, layer_name='first_conv')
    pool1 = MaxPool2D(pool_size=(3, 3), strides=2, padding='same', name='pool1')(conv1)
    
    conv = pool1
    for i in range(4):
        conv = make_layer(conv, block=block, block_num=block_num[i], kernel_num=kernel_num[i],
                          layer_name='layer' + str(i + 1))
    
    pool2 = GlobalAvgPool2D(name='globalavgpool')(conv)
    output_ = Dense(nclass, activation=activation, name='dense')(pool2)
    
    model = Model(inputs=input_, outputs=output_, name='ResNet')
    # model.summary()
    
    return model


# if __name__ == '__main__':
#     ResNet(input_shape=(224, 224, 3), nclass=2, net_name='resnet18')

