import tensorflow as tf
import glob
import os
import numpy as np
from PIL import Image
from shutil import copy, rmtree

# 将部分 visual_wake_words 中的图像还原成图片

num_classes = 2
batch_size = 1

input_data_dir = "./data_set/mini_vww"
output_data_dir = "./data_set/mini_vww_img"

train_background_dir = os.path.join(output_data_dir, 'train', 'background')
train_person_dir = os.path.join(output_data_dir, 'train', 'person')
val_background_dir = os.path.join(output_data_dir, 'val', 'background')
val_person_dir = os.path.join(output_data_dir, 'val', 'person')


def mk_file(file_path: str):
    if os.path.exists(file_path):
        # 如果文件夹存在，则先删除原文件夹在重新创建
        rmtree(file_path)
    os.makedirs(file_path)


def tf_record_num(file_path):
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
        num_samples_in_record = tf_record_num(record_path)
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
        # 'image/format':
        #     tf.io.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/class/label':
            tf.io.FixedLenFeature([], tf.int64, default_value=-1),
    }
    
    # Load one example
    # proto 是单个图像样本，是一串序列
    # parsed_features 是一个字典，key 是上面的特征，value 是 tensor
    # parsed_featrues 是单个图像样本的特征 字典
    parsed_features = tf.io.parse_single_example(proto, keys_to_features)
    
    # 源码中将 float64 转变为 int32，我们这是 int64，不知道需不需要转换，蛮试试
    label = tf.cast(parsed_features['image/class/label'], tf.int32)
    image = tf.io.decode_jpeg(parsed_features['image/encoded'], channels=3)
    image = tf.cast(image, dtype='float32') / 255.
    image = tf.image.resize(image, [parsed_features['image/height'], parsed_features['image/width']])
    
    label = tf.one_hot(label, num_classes)
    
    return image, label


def create_dataset(data_dir, dataset_type):
    # This works with arrays as well
    tfrecord_list = glob.glob("{}/{}.record-*".format(data_dir, dataset_type))
    dataset = tf.data.TFRecordDataset(tfrecord_list)
    
    dataset = dataset.prefetch(buffer_size=batch_size)
    
    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    # map 对数据集中的每一个 element 都作用了 map_func(在这里是 _parse_function)
    dataset = dataset.map(_parse_function, num_parallel_calls=8)
    
    # This dataset will go on forever
    # repeat(3)数据集就会被重复3次，没指定次数就会被无限重复
    # repeat 的作用是定义模型在每个 epoch 被训练几次，即步数 steps
    # 不指定具体的 epoch 数量，以防止迭代到最后一个 batch 时，出现 out of sequence
    dataset = dataset.repeat()
    
    # Set the number of datapoints you want to load and shuffle
    # shuffle 的大小应该大于 buffer_size
    # dataset = dataset.shuffle(buffer_size=(2 * batch_size))
    
    # Set the batchsize
    # dataset = dataset.apply(tf.data.Dataset.unbatch())
    # dataset = dataset.unbatch()
    dataset = dataset.batch(batch_size=batch_size)
    
    # Create an iterator
    # iterator = dataset.make_one_shot_iterator()
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    
    return iterator


# label[0][0] = 1. , 为 background, label[0][0] = 0. ，为person
# img_type 只能是 'train' 或 'val'
def data_handler(img_num, img_iterator, img_type):
    if img_type == 'train':
        background_dir = train_background_dir
        person_dir = train_person_dir
    elif img_type == 'val':
        background_dir = val_background_dir
        person_dir = val_person_dir
    else:
        print("type error")
        exit(-1)
    
    for idx in range(img_num):
        image, label = img_iterator.get_next()
        
        # 得到的数据格式与 generator 完全相同
        image = image.numpy()
        label = label.numpy()
        
        image = image[0] * 255
        image = image.astype(np.uint8)
        image = Image.fromarray(image)
        
        # label[0][0] == 1: background
        # label[0][0] == 0: person
        if label[0][0] == 1:
            image.save(os.path.join(background_dir, img_type + '_{}.jpg'.format(idx)))
        else:
            image.save(os.path.join(person_dir, img_type + '_{}.jpg'.format(idx)))
        
        print('image {} done'.format(idx))


mk_file(train_background_dir)
mk_file(train_person_dir)
mk_file(val_background_dir)
mk_file(val_person_dir)

total_num_train = get_samples_num(input_data_dir, 'train')
total_num_val = get_samples_num(input_data_dir, 'val')
total_num = total_num_val + total_num_train
print('data num waiting to be converted: {}'.format(total_num))

train_iterator = create_dataset(input_data_dir, 'val')
val_iterator = create_dataset(input_data_dir, 'train')

data_handler(total_num_train, train_iterator, 'train')
data_handler(total_num_val, val_iterator, 'val')


    



# 独热码测试

# label_0 = [0]
# label_1 = [1]
# label_0 = tf.one_hot(label_0, num_classes)
# label_1 = tf.one_hot(label_1, num_classes)
#
# print(label_0.numpy())
# print(label_1.numpy())

    
    