import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential


# 此函数用来保证每一层的通道数都可以被 8 整除
def _make_divisible(ch, divisor=8, min_ch=None):
    if min_ch is None:
        min_ch = divisor
    # // 意思是返回商的整数部分（向下取整）
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # 确保通道数下降幅度不会超过 10%
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class ConvBNReLU(layers.Layer):
    # 搭建标准卷积层，跟上 BN正则化 和 ReLu 激活函数，一套全打包
    def __init__(self, out_channel, kernel_size=3, stride=1, **kwargs):
        # 集成 layers.Layer 父类的方法，并在起基础上加了自己的方法
        super(ConvBNReLU, self).__init__(**kwargs)
        # Conv2D 参数讲解
        # filters: 卷积核个数，即输出通道数
        # kernel_size：卷积核大小
        # strides：步长
        # padding: 两个值，'valid'和'same'，'valid'表明不要 padding, 'same'表明在 input 周围均匀地填充 0 以使 output 的
        #          width/height 与 input 相同
        # use_bias: 是否使用偏差
        # name: 这一层的名字
        self.conv = layers.Conv2D(filters=out_channel, kernel_size=kernel_size, strides=stride,
                                  padding='SAME', use_bias=False, name='Conv2d')
        # BatchNormalization 参数讲解
        # momentum: 动量，与学习速率率成反比； batch_size 越大，momentum 应该越小
        # epsilon: 加到方差上的一个浮点数，以防止方差为0
        self.bn = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='BatchNorm')
        # ReLU 参数讲解
        # max_value: 最大激活值（激活后最多只能这么大）
        self.activation = layers.ReLU(max_value=6.0)
    
    # 为了能保存h5模型，必须要更新 init 中新加的参数
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'conv': self.conv,
            'bn': self.bn,
            'activation': self.activation,
        })
        return config
    
    def call(self, inputs, training=False):
        # 返回的 x 是一个张量或元素为张量的列表/元组
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = self.activation(x)
        return x


class InvertedResidual(layers.Layer):
    def __init__(self, in_channel, out_channel, stride, expand_ratio, **kwargs):
        super(InvertedResidual, self).__init__(**kwargs)
        self.hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel
        
        layer_list = []
        if expand_ratio != 1:
            # 1*1 pointwise conv
            layer_list.append(ConvBNReLU(out_channel=self.hidden_channel, kernel_size=1, name='expand'))
        
        layer_list.extend([
            # 3*3 depthwise conv
            layers.DepthwiseConv2D(kernel_size=3, padding='SAME', strides=stride,
                                   use_bias=False, name='depthwise'),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='depthwise/BatchNorm'),
            layers.ReLU(max_value=6.0),
            # 1*1 pointwise conv(linear)
            layers.Conv2D(filters=out_channel, kernel_size=1, strides=1,
                          padding="SAME", use_bias=False, name="project"),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='pointwise/BatchNorm')
        ])
        self.main_branch = Sequential(layer_list, name='expanded_conv')
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'hidden_channel': self.hidden_channel,
            'use_shortcut': self.use_shortcut,
        })
        return config
    
    def call(self, inputs, training=False):
        if self.use_shortcut:
            return inputs + self.main_branch(inputs, training=training)
        else:
            return self.main_branch(inputs)


def mobile_net_v2(im_height=224,
                  im_width=224,
                  num_classes=1000,
                  alpha=1.0,
                  divisor=8,
                  include_top=True):
    # 源代码中的 round_nearest(divisor) 都为8， 干脆不要了
    
    block = InvertedResidual
    
    # 输入 32 通道，输出 1280 通道是论文中规定的
    input_channel = _make_divisible(32 * alpha, divisor)
    last_channel = _make_divisible(1280 * alpha, divisor)
    inverted_residual_setting = [
        # t(expand_ratio), c(输出通道数), n(相同层重复次数), s(stride)
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1],
    ]
    
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype='float32')
    
    # conv1
    x = ConvBNReLU(input_channel, stride=2, name='Conv')(input_image)
    
    # building inverted residual residual blocks
    # 建立层层网络的目的就是要得到最后的权重
    # 所以不管是直接把卷积层列出来也好，还是像下面这样通过循环求最终权重也好
    # 反正最后得到的是一列权重值就行
    for index, (t, c, n, s) in enumerate(inverted_residual_setting):
        output_channel = _make_divisible(c * alpha, divisor)
        for i in range(n):
            stride = s if i == 0 else 1
            x = block(x.shape[-1],
                      output_channel,
                      stride,
                      expand_ratio=t)(x)
    x = ConvBNReLU(last_channel, kernel_size=1, name='Conv_1')(x)
    
    if include_top is True:
        # building classifier
        x = layers.GlobalAveragePooling2D()(x)  # pool + faltten
        x = layers.Dropout(rate=0.5)(x)
        x = layers.Dense(num_classes, name='Logits')(x)
        output = layers.Softmax()(x)
    else:
        output = x
    
    # 定义完 layers 后还要调用 Model 才算真正建立了模型
    model = Model(inputs=input_image, outputs=output)
    return model


if __name__ == '__main__':
    model = mobile_net_v2(num_classes=2)
    model.summary()
    officialModel = tf.keras.applications.mobilenet_v2.MobileNetV2(weights=None, classes=2)
    officialModel.summary()
    
    


