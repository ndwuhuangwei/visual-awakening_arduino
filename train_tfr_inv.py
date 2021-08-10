# DL import
import tensorflow as tf
from resnet_model import ResNet
# from MobileNetV2_model import mobile_net_v2
# from Involution_MobileNetV2 import MobileNetV2
# import tensorflow_model_optimization as tfmot

from micronet_model import MicroNet

# other import
import glob
# from tqdm import tqdm
import datetime
import time
import os

from resnet_seblock_model import ResNet_SE

# GPU configuration
gpu_device = 0
gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[gpu_device], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[gpu_device], True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

# ----------------- 训练之前必须检查的全部变量 ----------------- #

# 如果要更换模型的话，只需要将 train() 中的 model 换掉

# 超参数
# # if class_num is 2, True, otherwise False
# binary_flag = True
#
# #  if use 3090, set True; if use 2070, set True
# is_3090 = False
#
# # default in 3090
# mini_vww = False

# 通用训练参数
batch_size = 32
epochs = 50
num_classes = 2
rho = 224
alpha = 1

# data_set_name = "vww"
# mini_test_set_dir = "/home/amax/Desktop/zzh/VWW_Code/data_set/visual_wake_words/test"
#
# if mini_vww:
#     data_set_dir = "/home/amax/Desktop/zzh/VWW_Code/data_set/visual_wake_words/mini_vww"
# elif is_3090:
#     data_set_dir = os.path.join("/home/amax/Desktop/zzh/VWW_Code/data_set/visual_wake_words", data_set_name)
# else:
#     data_set_dir = os.path.join("/home/zzh/Desktop/TinyML/datasets/visual_wake_words", data_set_name)

# data_set_dir = './data_set/mini_vww'
data_set_dir = '../data_set/vww'

train_name = "train"
val_name = "val"

# pre_weights_name = 'mobilenet_v2_0.35_224'
# pre_weights_path = os.path.join('./pre_weights', pre_weights_name, pre_weights_name + '.ckpt')

# weights_name = "MobieNet_Model_{}.ckpt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
# weights_name_dir = "MobieNet_Model_{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
# weights_name_h5 = "MobieNet_Model_{}.h5".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
# weights_export_path = os.path.join('./weights', weights_name)
# weights_export_path_h5 = os.path.join('./weights', weights_name_h5)

# log_export_dir = "./log"

# 用于 shuffle, 大于等于数据集大小
buffer_size = 100000


# --------------------------- END --------------------------- #


def img_preprocess(img):
    img = img / 255.
    img = (img - 0.5) * 2.0
    return img


def tf_record_nun(file_path):
    # 统计某个 tfrecord 文件中的样本数目
    cnt = 0
    for record in tf.compat.v1.python_io.tf_record_iterator(file_path):
        cnt += 1
    return cnt


def get_samples_num(data_dir, record_name):
    # 功能：统计某个目录下 tfrecords 中的样本数目 (可以统计被分割成多个 tfrecords 文件的形式)
    # data_dir: 训练集或验证集的 tfrecords 文件的根目录
    total_num = 0
    tfrecord_path_list = glob.glob("{}/{}.record-*".format(data_dir, record_name))
    for record_path in tfrecord_path_list:
        print("counting {}".format(record_path))
        num_samples_in_record = tf_record_nun(record_path)
        total_num += num_samples_in_record
    
    return total_num


def _parse_function(proto):
    # define your tfrecord again. Remember that you saved your image as a string.
    # bytes 和 float 对应什么都是猜的
    keys_to_features = {
        'image/height':
            tf.io.FixedLenFeature([], tf.int64),
        'image/width':
            tf.io.FixedLenFeature([], tf.int64),
        'image/encoded':
            tf.io.FixedLenFeature((), tf.string, default_value=''),
        'image/class/label':
            tf.io.FixedLenFeature([], tf.int64, default_value=-1),
    }
    
    # Load one example
    # proto 是单个图像样本，是一串序列
    # parsed_features 是一个字典，key 是上面的特征，value 是 tensor
    # parsed_featrues 是单个图像样本的特征 字典
    parsed_features = tf.io.parse_single_example(proto, keys_to_features)
    
    # # Turn your saved image string into an array
    # # 转换后是 tensor
    # # parsed_features['image/encoded'] 是具体的数组/数字
    # image = tf.io.decode_raw(parsed_features['image/encoded'], tf.uint8)
    # height = parsed_features['image/height']
    # width = parsed_features['image/width']
    
    image = tf.io.decode_jpeg(parsed_features['image/encoded'], channels=3)
    image = tf.cast(image, dtype='float32') / 255.
    # image = tf.cast(image, dtype='float32')
    # image = img_preprocess(image)
    image = tf.image.resize(image, [rho, rho])
    
    # 源码中将 float64 转变为 int32，我们这是 int64，不知道需不需要转换，蛮试试
    label = tf.cast(parsed_features['image/class/label'], tf.int32)
    
    # # [height, width, channel]
    # image = tf.reshape(image, [height, width, 3])
    # image = tf.image.resize(image, [rho, rho])
    #
    # # 将图片归一化到 -1-1 的范围
    # image = img_preprocess(image)
    
    # 标签要变为 one-hot 编码格式
    label = tf.one_hot(label, num_classes)
    
    return image, label


def create_dataset(dataset_dir, dataset_type):
    # This works with arrays as well
    tfrecord_list = glob.glob("{}/{}.record-*".format(dataset_dir, dataset_type))
    dataset = tf.data.TFRecordDataset(tfrecord_list)
    
    dataset = dataset.prefetch(buffer_size=batch_size)
    
    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    # map 对数据集中的每一个 element 都作用了 map_func(在这里是 _parse_function)
    dataset = dataset.map(_parse_function, num_parallel_calls=16)
    
    # This dataset will go on forever
    # repeat(3)数据集就会被重复3次，没指定次数就会被无限重复
    # repeat 的作用是定义模型在每个 epoch 被训练几次，即步数 steps
    # 不指定具体的 epoch 数量，以防止迭代到最后一个 batch 时，出现 out of sequence
    dataset = dataset.repeat()
    
    # Set the number of datapoints you want to load and shuffle
    # shuffle 的大小应该大于 buffer_size
    dataset = dataset.shuffle(buffer_size=(2 * batch_size))
    
    # Set the batchsize
    dataset = dataset.batch(batch_size=batch_size)
    
    # Create an iterator
    # iterator = dataset.make_one_shot_iterator()
    # 如果用model.fit进行训练的话就可以不手动生成迭代器
    # iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    
    return dataset


def train():
    ###########################
    # 准备数据
    ###########################
    train_dataset = create_dataset(data_set_dir, train_name)
    val_dataset = create_dataset(data_set_dir, val_name)
    # test_dataset = create_dataset(mini_test_set_dir, val_name)
    
    total_train_num = get_samples_num(data_set_dir, train_name)
    total_val_num = get_samples_num(data_set_dir, val_name)
    # total_test_num = get_samples_num(mini_test_set_dir, val_name)
    
    ###########################
    # 定义网络
    ###########################
    
    # model = tf.keras.applications.resnet50.ResNet50(include_top=True, weights=None, classes=2)
    # model = ResNet(input_shape=[rho, rho, 3], nclass=num_classes, activation='sigmoid', net_name='resnet26')
    # model = ResNet_SE(input_shape=[rho, rho, 3], nclass=num_classes, activation='sigmoid', net_name='resnet26')
    # model = tf.keras.applications.mobilenet_v2.MobileNetV2(alpha=1.0, weights=None, classes=num_classes)
    
    model = MicroNet(input_shape=(160, 160, 3), alpha=1.0, weights=None, classes=2)
    
    model.trainable = True
    
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    model.summary()
    
    ######################################################################
    # 普通训练
    ######################################################################
    
    # 保存效果最好的模型
    weights_name = "Micronet_VWW_{}.ckpt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    weights_export_path = os.path.join(os.getcwd(), 'weights', weights_name)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=weights_export_path,
                                                             monitor='val_accuracy',
                                                             verbose=1,
                                                             save_best_only=True,
                                                             save_weights_only=True,
                                                             mode='max')
    
    # 为 tensorboard 准备目录
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    current_dir = os.getcwd()
    tb_name = 'MicroNet_VWW_'
    # 在win10下，为tensorboard准备的路径不能含有斜杆，必须用os.path.join进行拼接
    log_dir = os.path.join(current_dir, 'train_log', tb_name + current_time)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    callbacks_list = [checkpoint_callback, tensorboard_callback]
    
    # 是数据集的话只需要将数据集赋给x, 如果是图像序列的话x核y分别赋 图像序列 和 标签序列
    # 如果传入数据集，不能自定义 batch_size 和 validation_rate
    # 必须定义 steps_per_epoch
    model.fit(x=train_dataset,
              epochs=epochs,
              verbose=1,
              callbacks=callbacks_list,
              validation_data=val_dataset,
              steps_per_epoch=total_train_num // batch_size,
              validation_steps=total_val_num // batch_size)

    # baseline_model_acc = model.evaluate(x=test_dataset, steps=total_test_num // batch_size)
    
    # clear the GPU memory, otherwise OOM when q train begins
    tf.keras.backend.clear_session()
    
    ######################################################################
    # 量化感知训练开始
    ######################################################################

    # quant_model = tfmot.quantization.keras.quantize_model(model)
    
    # quant_model.load_weights(weights_export_path)
    # quant_model.compile(optimizer=tf.keras.optimizers.Adam(),
    #                    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    #                    metrics=['accuracy'])
    # quant_model.summary()

    # weights_name = "quant_Model_{}.ckpt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    # quant_weights_export_path = os.path.join(os.getcwd(), 'weights', weights_name)
    # quant_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=quant_weights_export_path,
    #                                                               monitor='val_accuracy',
    #                                                               verbose=1,
    #                                                               save_best_only=True,
    #                                                               save_weights_only=True,
    #                                                               mode='max')
    
    # quant训练单独用一个tensorboard
    # current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # current_dir = os.getcwd()
    # 在win10下，为tensorboard准备的路径不能含有斜杆，必须用os.path.join进行拼接
    # quant_log_dir = os.path.join(current_dir, 'train_log', current_time)
    # quant_tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=quant_log_dir, histogram_freq=1)

    # quant_callbacks_list = [quant_checkpoint_callback, quant_tensorboard_callback]

    # quant_model.fit(x=train_dataset,
    #                epochs=epochs,
    #                verbose=1,
    #                callbacks=quant_callbacks_list,
    #                validation_data=val_dataset,
    #                steps_per_epoch=total_train_num // quant_batch_size,
    #                validation_steps=total_val_num // quant_batch_size)
    
    ######################################################################
    # 比较两种模型的预测精读
    ######################################################################
    # baseline_model_acc = model.evaluate(x=test_dataset, steps=total_test_num // batch_size)
    # quant_aware_model_acc = quant_model.evaluate(x=test_dataset, steps=total_test_num // quant_batch_size)
    # print('Baseline test accuracy:', baseline_model_acc)
    # print('Quant test accuracy:', quant_aware_model_acc)
    
    ######################################################################
    # 得到真正的量化模型
    ######################################################################
    # 量化感知模型参数的数据类型还是 float32
    # converter = tf.lite.TFLiteConverter.from_keras_model(quant_model)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # quantized_tflite_model = converter.convert()

    # if not os.path.exists('./tflite_models'):
    #    os.mkdir('./tflite_models')

    # quant_tflite_model = "quant_tfliteModel_{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    # tflite_model_path = os.path.join('./tflite_models', quant_tflite_model)
    # with open(tflite_model_path, 'wb') as f:
    #    f.write(quantized_tflite_model)


if __name__ == '__main__':
    start_time = time.time()
    # test_model = train()
    train()
    end_time = time.time()
    print('train time cost: {}s'.format(end_time - start_time))
    # print('test begin')
    #
    # test_dir = os.path.join(os.getcwd(), 'data_set', 'visual_wake_words', 'test')
    # assert os.path.exists(test_dir), "can't find {}".format(test_dir)
    #
    # test_result = test_predict(model=test_model, test_dir=test_dir, weights_load_path=weights_export_path)
    # print("test_result :{}".format(test_result))
