import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import glob
from tqdm import tqdm
import datetime
import time
import os
from Involution_MobileNetV2 import MobileNetV2

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


# 超参数
# 范围 [0.35, 0.50, 0.75, 1.0, 1.3, 1.4]
alpha = 1
# 最小 32
rho = 224

# 通用训练参数
batch_size = 64  # 3090上跑的话，可设置为 >=512
epochs = 50
num_classes = 2
shuffer_size = 10*batch_size

data_set_name = "vww"
data_set_dir = os.path.join("/home/amax/Desktop/zzh/VWW_Code/data_set/visual_wake_words", data_set_name)

train_name = "train"
val_name = "val"

weights_name = "MobieNet_Model_{}.ckpt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
weights_name_dir = "MobieNet_Model_{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
weights_name_h5 = "MobieNet_Model_{}.h5".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
weights_export_path = os.path.join('./weights', weights_name)
# weights_export_path_h5 = os.path.join('./weights', weights_name_h5)

log_export_dir = "./log"
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
        # print("counting {}".format(record_path))
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
    dataset = dataset.map(_parse_function, num_parallel_calls=10)

    # This dataset will go on forever
    # repeat(3)数据集就会被重复3次，没指定次数就会被无限重复
    # repeat 的作用是定义模型在每个 epoch 被训练几次，即步数 steps
    # 不指定具体的 epoch 数量，以防止迭代到最后一个 batch 时，出现 out of sequence
    dataset = dataset.repeat()

    # Set the number of datapoints you want to load and shuffle
    # shuffle 的大小应该大于 buffer_size
    dataset = dataset.shuffle(buffer_size=shuffer_size)

    # Set the batchsize
    dataset = dataset.batch(batch_size=batch_size)

    # Create an iterator
    # iterator = dataset.make_one_shot_iterator()
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)

    return iterator

def train():
    train_iterator = create_dataset(data_set_dir, train_name)
    val_iterator = create_dataset(data_set_dir, val_name)

    total_train_num = get_samples_num(data_set_dir, train_name)
    total_val_num = get_samples_num(data_set_dir, val_name)


    ######################################
    #              加载模型
    ######################################
    model = MobileNetV2(include_top=True, weights=None, alpha=alpha,
                                input_shape=(rho, rho, 3), classes=num_classes,
                                classifier_activation='softmax', dropout=0.5)
    model.trainable = True
    model.summary()

    model_name = '_mobiletv_rho224_alpha1_inv'

    # 为 tensorboard 准备目录
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = './train_log/' + current_time + model_name + '/train'
    val_log_dir = './train_log/' + current_time + model_name +'/val'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    # 设置训练参数
    # 首先选择损失器和优化器
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(
                                initial_learning_rate=0.01, decay_steps=total_val_num // batch_size, decay_rate=0.98)
    optimizer = tf.keras.optimizers.Adam(learning_rate=exponential_decay)

    # 创建中间变量
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.CategoricalAccuracy(name='val_accuracy')

    # 定义训练和测试代码
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)
        grads = tape.gradient(loss, model.trainable_variables)

        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def val_step(images, labels):
        predictions = model(images, training=False)
        loss = loss_object(labels, predictions)

        val_loss(loss)
        val_accuracy(labels, predictions)

    # 正式开始训练
    best_val_acc = 0
    for epoch in range(epochs):
        # 为训练进度设置进度条，更好看
        train_bar = tqdm(range(total_train_num // batch_size))
        # 进度条每前进一格，就拿一个 batch 的数据
        for step in train_bar:
            train_images, train_labels = train_iterator.get_next()
            # print("train data load done")
            train_images = train_images.numpy()
            train_labels = train_labels.numpy()

            train_step(train_images, train_labels)

            train_bar.desc = "train epoch[{}/{}] Loss:{:.3f}, acc{:.3f}".format(epoch + 1,
                                                                                epochs,
                                                                                train_loss.result(),
                                                                                train_accuracy.result())

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

        # val 一模一样
        val_bar = tqdm(range(total_val_num // batch_size))
        for step in val_bar:
            val_images, val_labels = val_iterator.get_next()
            # print("val data load done")
            val_images = val_images.numpy()
            val_labels = val_labels.numpy()

            val_step(val_images, val_labels)

            val_bar.desc = "val epoch[{}/{}] Loss:{:.3f}, acc{:.3f}".format(epoch + 1,
                                                                            epochs,
                                                                            val_loss.result(),
                                                                            val_accuracy.result())

        with val_summary_writer.as_default():
            tf.summary.scalar('loss', val_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', val_accuracy.result(), step=epoch)

        # only save best weights
        if val_accuracy.result() > best_val_acc:
            best_val_acc = val_accuracy.result()
            model.save_weights(weights_export_path, save_format="tf")
            # model.save(weights_export_path_h5)

        # 训练完清空历史数据
        train_loss.reset_states()  # clear history info
        train_accuracy.reset_states()  # clear history info
        val_loss.reset_states()  # clear history info
        val_accuracy.reset_states()  # clear history info

    # 方便测试
    return model


if __name__ == '__main__':
    start_time = time.time()
    test_model = train()
    end_time = time.time()
    print('train time cost: {}s'.format(end_time - start_time))
    # print('test begin')
    #
    # test_dir = os.path.join(os.getcwd(), 'data_set', 'visual_wake_words', 'test')
    # assert os.path.exists(test_dir), "can't find {}".format(test_dir)
    #
    # test_result = test_predict(model=test_model, test_dir=test_dir, weights_load_path=weights_export_path)
    # print("test_result :{}".format(test_result))
