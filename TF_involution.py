# This is a simple implimentation of Involution with Tensorflow2
# "Involution- Inverting the Inherence of Convolution for Visual Recognition"

import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers



'''
tf.keras.layers.Conv2D(
    filters, kernel_size, strides=(1, 1), padding='valid',
    data_format=None, dilation_rate=(1, 1), groups=1, activation=None,
    use_bias=True, kernel_initializer='glorot_uniform',
    bias_initializer='zeros', kernel_regularizer=None,
    bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
    bias_constraint=None, **kwargs
)                
'''

# 从 空间 和 通道 两个维度来考虑，没有本质差异，只是算力的不同分配方式，对空间进行的各种操作转移到了通道这一维度
# convolution 空间不变性 与 通道特异性
# involution  空间特异性 与 通道不变性


class Involution(layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=1, padding='same', dilation_rate=1, groups=1, reduce_ratio=1):
        super(Involution, self).__init__()
        self.filters = filters  # the output channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.groups = groups  # the number of kernels
        self.reduce_ratio = reduce_ratio
        self.reduce_mapping = tf.keras.Sequential(
            [
                layers.Conv2D(filters // reduce_ratio, 1, padding=padding),
                layers.BatchNormalization(),
                layers.ReLU(6.),
                # layers.Activation('relu'),
            ]
        )
        self.span_mapping = layers.Conv2D(kernel_size * kernel_size * groups, 1, padding=padding)

        self.initial_mapping = layers.Conv2D(self.filters, 1, padding=padding)
        if strides > 1:
            # self.o_mapping = layers.AveragePooling2D(strides)
            self.o_mapping = layers.MaxPooling2D(strides)

    def call(self, x):
        # generate the involution kernel adapting to the input feature map
        inv_kernel = x if self.strides == 1 else self.o_mapping(x)  # 池化 [batch, H, W, C]         ->[batch, H//s, W//s, C], s=strides
        inv_kernel = self.reduce_mapping(inv_kernel)             # 通道缩减 [batch, H//s, W//s, C]   ->[batch, H//s, W//s, filters//r], r=reduce_ratio, filters=C(in paper)
        inv_kernel = self.span_mapping(inv_kernel)         # 通道变换 [batch, H//s, W//s, filters//r]->[batch, H//s, W//s, K*K*G], G=groups
        _, h, w, c = K.int_shape(inv_kernel)  # [batch, H//s, W//s, K*K*G]
        # reshape [batch, H//s, W//s, K*K*G] -> [batch, H//s, W//s, G, K*K] -> [batch, H//s, W//s, G, 1, K*K]
        inv_kernel = K.expand_dims(K.reshape(inv_kernel, (-1, h, w, self.groups, self.kernel_size * self.kernel_size)), axis=4)

        # [batch, H, W, C] -> [batch, H, W, filters] -> [batch, H//s, W//s, kernel*kernel*filters] (s=strides=1,dilation_rate=1,padding='SAME')
        out = tf.image.extract_patches(images=x if self.filters == c else self.initial_mapping(x),
                                       sizes=[1, self.kernel_size, self.kernel_size, 1],
                                       strides=[1, self.strides, self.strides, 1],
                                       rates=[1, self.dilation_rate, self.dilation_rate, 1],
                                       padding="SAME" if self.padding == 'same' else "VALID")
        # 将每个pixel对应邻域的特征向量分为G组(对应G个kernel)，每组包含filters//G个通道
        # [batch, H//s, W//s, kernel*kernel*filters] -> [batch, H//s, W//s, G, filters//G, kernel*kernel]
        out = K.reshape(out, (-1, h, w, self.groups, self.filters // self.groups, self.kernel_size * self.kernel_size))

        # 1.element-wise product
        # [batch, H//s, W//s, G, 1, K*K] * [batch, H//s, W//s, G, filters//G, kernel*kernel] -> [batch, H//s, W//s, G, filters//G, kernel*kernel]
        # 2.sum
        # [batch, H//s, W//s, G, filters//G, kernel*kernel] -> [batch, H//s, W//s, G, filters//G, 1] -> [batch, H//s, W//s, G, filters//G]
        out = K.sum(inv_kernel * out, axis=-1)
        # [batch, H//s, W//s, G, filters//G] -> [batch, H//s, W//s, filters]
        out = K.reshape(out, (-1, h, w, self.filters))
        return out


# class Involution(tf.keras.layers.Layer):
#     # 输入一个 B,H,W,C 的特征图
#     # 输出一个 B,(H-K+2*padding)/S+1, (W-K+2*padding)/S+1, C 的特征图
#     # 与convolution 相比，
#     def __init__(self, channel, group_number, kernel_size, stride, reduction_ratio):
#         # channel: 输入特征图的 channel(因为 involution channel不变，所以输出也是这个channel)
#         super().__init__()
#         # The assert makes sure that the user knows about the
#         # reduction size. We cannot have 0 filters in Conv2D.
#         assert reduction_ratio <= channel, print("Reduction ration must be less than or equal to channel size")
#
#         self.channel = channel
#         self.group_number = group_number
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.reduction_ratio = reduction_ratio
#
#         # 为了使生成的kernel的shape与patches对应，H,W必须一致
#         # 可以通过池化才达成目的
#         self.o_weights = tf.keras.layers.AveragePooling2D(
#             pool_size=self.kernel_size,
#             strides=self.stride,
#             padding="same") if self.stride > 1 else tf.identity
#
#         # key part: 生成 kernal
#         # kernal size 为1，等于全连接层
#         self.kernel_gen = tf.keras.Sequential([
#             # 先减少 channel 的数量
#             tf.keras.layers.Conv2D(
#                 filters=self.channel // self.reduction_ratio,
#                 kernel_size=1),
#             tf.keras.layers.BatchNormalization(),
#             tf.keras.layers.ReLU(),
#             tf.keras.layers.Conv2D(
#                 # 把 channel 的数量变成 K*K*G
#                 # 虽然不是本意，但是这也是在减小通道数量
#                 # 因为神经网络一般是纺锤结构，每层的参数量先不断上升，然后再下降
#                 # 对于中间层来说，K*K*G 远远达不到它们的参数量
#                 filters=self.kernel_size * self.kernel_size * self.group_number,
#                 kernel_size=1)
#         ])
#
#     def get_config(self):
#         config = super().get_config().copy()
#         config.update({
#             'channel': self.channel,
#             'group_number': self.group_number,
#             'kernal_size': self.kernel_size,
#             'stride': self.stride,
#             'reduction_ratio': self.reduction_ratio,
#             'o_weights': self.o_weights,
#             'kernel_gen': self.kernel_gen,
#         })
#         return config
#
#     def call(self, x, training=False):
#         # involution执行过程中要生成两个东西，一个是由原特征图变换而来的 pathes; 一个是由原特征图变换而来的
#
#         # generate the patches
#
#         _, H, W, C = x.shape
#         # Extract input feature blocks
#         # 从输入的图像矩阵中抽出一些 pathes
#         # 输出是这些 pathes 的组合
#         # 其实也可以看作是一种降采样
#         #
#         # sizes 和 strides 只有中间两个数可以变，首尾两个必须都是1
#         #
#         # 这个api的输出 本来就会使最后一个纬度变成 C*kernel_size*kernel_size
#         unfolded_x = tf.image.extract_patches(
#             images=x,
#             sizes=[1, self.kernel_size, self.kernel_size, 1],
#             strides=[1, self.stride, self.stride, 1],
#             rates=[1, 1, 1, 1],
#             padding="SAME")
#         # B, (H-K+2*padding)/S+1, (W-K+2*padding)/s+1, K*K*C
#         # 输入图像是224的情况下，kernal_size=3, strides=1, H,W正好不变
#
#         # 如果 C//group_number 不能整除的话结果要补上被忽略的值
#         # 设 (X*G + Y) // G = X, 其中 Y < G
#         # 那么显然 X * G != X*G + Y
#         # 那么我们应该怎么做？
#         # 限制 G 的值，G 不是任意取的，必须得能整除
#         # C 为3，G 只能是1或3
#         unfolded_x = tf.keras.layers.Reshape(
#             # 因为卷积时有 padding 所以要 +2
#             target_shape=((H-self.kernel_size+2)//self.stride + 1,
#                           (W-self.kernel_size+2)//self.stride + 1,
#                           self.kernel_size * self.kernel_size,
#                           C // self.group_number,
#                           self.group_number))(unfolded_x)  # B, (H-K+2*padding)/S+1, (W-K+2*padding)/s+1, K*K, C//G, G
#         # involution的filter建立完成
#         # 原特征图被分成 G 组，每组有 C//G 个 (B, (H-K+2*padding)/S+1, (W-K+2*padding)/s+1, K*K)
#         # 相比原特征图，新特征图多了 K*K 倍的像素点，这都是 tf.image.extract_patches 这个API带来的
#
#         # generate the kernel
#
#         # 首先降采样, 目的是使每个通道的 spatial dimensions与 patches 相同
#         kernel_inp = self.o_weights(x)  # B, (H-K+2*padding)/S+1, (W-K+2*padding)/s+1, C
#         # 将通道数变为 kernel_size * kernel_size * G
#         kernel = self.kernel_gen(kernel_inp)  # B, (H-K+2*padding)/S+1, (W-K+2*padding)/s+1, K*K*G
#         # 然后将特征图的shape 四维改六维，与patches 一一对应
#         kernel = tf.keras.layers.Reshape(
#             target_shape=((H-self.kernel_size+2)//self.stride + 1,
#                           (W-self.kernel_size+2)//self.stride + 1,
#                           self.kernel_size * self.kernel_size,
#                           1,
#                           self.group_number))(kernel)  # B, (H-K+2*padding)/S+1, (W-K+2*padding)/s+1, K*K, 1, G
#         # 同样被分为G组，每组 1 个 (B, H, W, self.kernel_size * self.kernel_size)
#
#         # 因此可以看出，每 C//G 个 shape 为 (B, (H-K+2*padding)/S+1, (W-K+2*padding)/s+1, K*K) 的 patch 对应 1 个 shape为
#         # (B, H, W, self.kernel_size * self.kernel_size)的 kernal
#
#         # Multiply-Add op
#         # multiply 是矩阵元素对应相乘
#         # 如果两个参数的shape种，一个为1，另一个大于1，那么最后输出大的那个纬度
#         out = tf.math.multiply(kernel, unfolded_x)  # B, (H-K+2*padding)/S+1, (W-K+2*padding)/s+1, K*K, C//G, G
#         # patches和kernal 相乘后产生 G 组特征图，每组 C//G 个 (B, (H-K+2*padding)/S+1, (W-K+2*padding)/s+1, K*K)
#
#         # axis代表求和的纬度，从0开始，axis=3代表第四个纬度，即 K*K，这个纬度求和后消失
#         out = tf.math.reduce_sum(out, axis=3)  # B, (H-K+2*padding)/S+1, (W-K+2*padding)/s+1, C//G, G
#         # 每组变成 C//G 个（B, (H-K+2*padding)/S+1, (W-K+2*padding)/s+1）矩阵
#
#         # 生成新图，
#         out = tf.keras.layers.Reshape(target_shape=((H-self.kernel_size+2)//self.stride + 1,
#                                                     (W-self.kernel_size+2)//self.stride + 1,
#                                                     C))(out)  # B, (H-K+2*padding)/S+1, (W-K+2*padding)/s+1, C
#         return out


# if __name__ == "__main__":
#     # invo = Involution2D(filters=3, kernel_size=3)
#     # x = tf.ones([1, 10, 10, 3])
#     # out = invo.call(x)

