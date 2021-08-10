import os
import json
import glob
import numpy as np

# from PIL import Image
# import matplotlib.pyplot as plt
import tensorflow as tf
from Involution_MobileNetV2_backup import Full_InvNet
from Involution_MobileNetV2_copy1 import MobileNetV2
from micronet_model import MicroNet
# from model_v2 import MobileNetV2

import time

alpha = 1
rho = 224
num_classes = 2
img_dir = "../data_set/mini_vww_img"
# img_path = "./test_data/test2.jpg"
# weights_path = './weights/Model_inv2_minivww_20210724-172945.ckpt'
weights_path = './weights/Model_MicroNet_VWW_20210802-133646.ckpt'


def load_model():
    time_1 = time.time()
    
    model = MicroNet(input_shape=(rho, rho, 3), alpha=1.0, weights=None, classes=2)
    # model = MobileNetV2(include_top=True, weights=None, alpha=alpha,
    #                     input_shape=(rho, rho, 3), classes=num_classes,
    #                     classifier_activation='sigmoid', dropout=0.5)
    # model = tf.keras.Sequential([feature,
    #                              tf.keras.layers.GlobalAvgPool2D(),
    #                              tf.keras.layers.Dropout(rate=0.5),
    #                              tf.keras.layers.Dense(num_classes),
    #                              tf.keras.layers.Softmax()])
    # weights_path = './save_weights/resMobileNetV2.ckpt'
    # model = tf.keras.applications.mobilenet_v2.MobileNetV2(alpha=1.0, weights=None, classes=num_classes)
    assert len(glob.glob(weights_path + "*")), "cannot find {}".format(weights_path)
    model.load_weights(weights_path).expect_partial()
    
    time_2 = time.time()
    # print('加载模型耗时：', str(time_2 - time_1), 's')
    
    return model, (time_2 - time_1)


# 预测一个文件夹的图片
'''
def predict_img(my_img_dir, my_model):
    # create model
    # feature = MobileNetV2(include_top=False)
    # time_1 = time.time()
    # feature = MobileNetV2(num_classes=num_classes, include_top=False)
    # model = tf.keras.Sequential([feature,
    #                              tf.keras.layers.GlobalAvgPool2D(),
    #                              tf.keras.layers.Dropout(rate=0.5),
    #                              tf.keras.layers.Dense(num_classes),
    #                              tf.keras.layers.Softmax()])
    # # weights_path = './save_weights/resMobileNetV2.ckpt'
    # time_2 = time.time()
    # print('加载模型耗时：', str(time_2 - time_1), 's')
    #
    # assert len(glob.glob(weights_path+"*")), "cannot find {}".format(weights_path)
    # model.load_weights(weights_path)

    for test_img in os.listdir(my_img_dir):
        print(test_img)
        img_path = os.path.join(my_img_dir, test_img)

        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path)
        # resize image to 224x224
        img = img.resize((im_width, im_height))
        plt.imshow(img)

        # scaling pixel value to (-1,1)
        img = np.array(img).astype(np.float32)
        img = ((img / 255.) - 0.5) * 2.0

        # Add the image to a batch where it's the only member.
        img = (np.expand_dims(img, 0))

        # read class_indict
        json_path = './class_indices.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

        json_file = open(json_path, "r")
        class_indict = json.load(json_file)

        # result表示预测正确概率, 因为有两个分类，所以是一个有两个元素的列表，元素值分别表示是class_indices中此索引所表示分类的概率
        # 比如现在class_indeices中 两个索引 0: Masked; 1: unMasked
        # 那么result = [0.8XX, 0.1XX] 分别为 Masked 的概率 和 unMasked的概率
        result = np.squeeze(my_model.predict(img))
        # result = np.squeeze(model(img, training=False))
        # print(result)

        # numpy.argmax（result）会返回reuslt中最大值的索引，在这里返回0
        predict_class = np.argmax(result)

        # 这里解释了为什么需要class_indices
        # 因为DNN输出的只是一个表示概率的列表，它不会直接输出这张属于哪个分类，但是它会按照class_indices中的索引顺序来排列概率
        # 所以我们要做的就是在输出的列表中找到概率最大值的索引，然后去class_indices中找这个索引代表哪个分类
        print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_class)],
                                                     result[predict_class])
        # plt.title(print_res)
        print(print_res)
        # plt.show()
'''


# 预测单张图片，摄像头读到直接预测完扔了，不用存在本地


def predict_img(img, my_model):
    # for test_img in os.listdir(my_img_dir):
    #     print(test_img)
    #     img_path = os.path.join(my_img_dir, test_img)
    #
    #     assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    #     img = Image.open(img_path)
    
    # resize image to 224x224
    img = img.resize((rho, rho))
    # plt.imshow(img)
    
    # scaling pixel value to (-1,1)
    img = np.array(img).astype(np.float32)
    img = ((img / 255.) - 0.5) * 2.0
    
    # Add the image to a batch where it's the only member.
    img = (np.expand_dims(img, 0))
    
    # read class_indict
    # json_path = './class_indices.json'
    # assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    
    ## json_file = open(json_path, "r")
    ## class_indict = json.load(json_file)
    
    # result表示预测正确概率, 因为有两个分类，所以是一个有两个元素的列表，元素值分别表示是class_indices中此索引所表示分类的概率
    # 比如现在class_indeices中 两个索引 0: Masked; 1: unMasked
    # 那么result = [0.8XX, 0.1XX] 分别为 Masked 的概率 和 unMasked的概率
    result = np.squeeze(my_model.predict(img))
    # result = np.squeeze(model(img, training=False))
    # print(result)
    
    # numpy.argmax（result）会返回reuslt中最大值的索引，在这里返回0
    predict_class_index = np.argmax(result)
    
    # 这里解释了为什么需要class_indices
    # 因为DNN输出的只是一个表示概率的列表，它不会直接输出这张属于哪个分类，但是它会按照class_indices中的索引顺序来排列概率
    # 所以我们要做的就是在输出的列表中找到概率最大值的索引，然后去class_indices中找这个索引代表哪个分类
    # print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_class)],
    #                                              result[predict_class])
    # plt.title(print_res)
    # print(print_res)
    # plt.show()
    
    # result_class = class_indict[str(predict_class)]
    # result_class = predict_class_index
    if predict_class_index == 0:
        result_class = 'person'
    elif predict_class_index == 1:
        result_class = 'background'
    
    result_prob = result[predict_class_index]
    
    return result_class, result_prob


def main():
    model, cost_time = load_model()
    print('加载模型耗时：', str(cost_time), 's')
    # predict_img(my_img_dir=img_dir, my_model=model)


if __name__ == '__main__':
    main()
