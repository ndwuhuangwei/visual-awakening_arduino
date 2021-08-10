# visual-awakening_arduino
An internship project for inlab, ZJU. A visual awakening system based on involution and arduino. 

# 文件夹架构

## daily log

此文件夹存放了一些重要工作的日志，全部都是md格式

## tflite_models

此文件夹存放量化后的tflite模型

## train_log

此文件夹存放训练日志，因为文件太大的原因，github上没有

## weights

存放保存的模型权重，因为文件太大的原因，github上没有

## 根目录 py 脚本

> Involution_MobileNetV2.py

> Involution_MobileNetV2_backup.py
 
> Involution_MobileNetV2_copy1.py

这三个文件存放不同版本的involuiton 网络

> micronet_inv_model.py

> micronet_model.py

这两个文件存放micronet的involution版本和cnn版本

> MobileNetV2_model.py

> official_mobilenetv2_model.py

此文件用于存放 MobileNetV2 网络，其中offcial开头的是tensorflow 的 github中的源码，另一个是自己写的，网络结构一样

> predict.py 

用于模型预测的代码，还有一个同名的 copy 文件是备份

> qt_app.py

qt程序源码，还有一个同名的 copy 文件是备份

> resnet_model.py

resnet 模型代码

> resnet_seblock_model.py

加入SELayer 后的resnet模型代码

> TF_involution.py

involution算子实现代码

> tflite_convert.py

将普通模型转换为 tflite 模型

> tfrecord2image.py

将 tfrecord 文件转换为 图像

> train_tfr_XXX.py

有很多 train_tfr_ 开头的脚本文件。这些都是使用 TFRecord 文件进行训练的代码，架构相同，只是使用的模型不一样，为了方便，就复制了很多训练脚本，并且一些训练脚本进行了加强，比如 gpus 后缀的训练脚本加入了并行训练，aug后缀的训练脚本加入了数据增强。



