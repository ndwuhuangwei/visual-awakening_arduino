import os
import json
import glob
import numpy as np
import cv2

# from PIL import Image
# import matplotlib.pyplot as plt
import tensorflow as tf
# from Involution_MobileNetV2_backup import Full_InvNet
# from Involution_MobileNetV2_copy1 import MobileNetV2
from Involution_MobileNetV2 import MobileNetV2_Inv
from micronet_model import MicroNet
# from model_v2 import MobileNetV2

import time

alpha = 1
rho = 224
num_classes = 2
img_dir = "../data_set/mini_vww_img"
# img_path = "./test_data/test2.jpg"
# weights_path = './weights/Model_inv2_minivww_20210724-172945.ckpt'
weights_dir = './weights/Model_MicroNet_VWW_20210802-133646'
# weights_dir_name = 'Model_MicroNet_VWW_20210802-133646'

'''获取文件的大小,结果保留两位小数，单位为MB'''


def get_FileSize(filePath):
    fsize = os.path.getsize(filePath)
    fsize = fsize / float(1024 * 1024)
    return round(fsize, 2)


def get_CkptSzie(path):
    ckpt_file_list = glob.glob("{}.*".format(path))
    size = 0
    for i in ckpt_file_list:
        size += get_FileSize(i)
    return size
    

def get_params(model):
    trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    count = trainable_count + non_trainable_count
    return count


def face_detect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_detector = cv2.CascadeClassifier("C:/Users/Whw03/Anaconda3/envs/github_test/Library/etc/haarcascades/"
                                          "haarcascade_frontalface_default.xml")
    # 最后一个参数设为1，会有两个框，faces会是一个二维列表，其中第一个列表元素代表的框是正确的
    # 设为2，只会留下不正确的偏小的那个框
    faces = face_detector.detectMultiScale(gray, 1.1, 1)
    # print(faces)
    # useful_face = faces[:1]
    # x, y为左上角，x为横轴，y为纵轴，原点在整张图片的左上角
    # for x, y, w, h in faces[:1]:
    #     cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    # cv2.imshow("result", image)
    # print(len(faces))
    if len(faces):
        xi = faces[0][0]
        yi = faces[0][1]
        wi = faces[0][2]
        hi = faces[0][3]
        return int(xi), int(yi), int(wi), int(hi)
    else:
        return 0, 0, 0, 0


def load_model(weights_dir, model_type):
    weights_name = weights_dir.split("/")[-1]
    weights_path = os.path.join(weights_dir, weights_name+'.ckpt')
    
    time_1 = time.time()
    
    if model_type == 'MicroNet':
        model = MicroNet(input_shape=(rho, rho, 3), alpha=1.0, weights=None, classes=2, classifier_activation='sigmoid')
    elif model_type == 'MobileNetV2':
        model = tf.keras.applications.mobilenet_v2.MobileNetV2(alpha=1.0, weights=None, classes=num_classes,
                                                               classifier_activation='sigmoid')
    elif model_type == 'MobileNetV2_Inv':
        model = MobileNetV2_Inv(include_top=True, weights=None, alpha=1,
                                input_shape=(rho, rho, 3), classes=2,
                                classifier_activation='sigmoid', dropout=0.5)
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
    
    return model, (time_2 - time_1), get_CkptSzie(weights_path), get_params(model)


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
        result_class = 'background'
    elif predict_class_index == 1:
        result_class = 'person'
    
    result_prob = result[predict_class_index]
    
    return result_class, result_prob


def main():
    model, cost_time, model_size, model_params = load_model(weights_dir)
    print('加载模型耗时：', str(cost_time), 's')
    print('加载模型耗时：', model_size, 'MB')
    # predict_img(my_img_dir=img_dir, my_model=model)


if __name__ == '__main__':
    main()
