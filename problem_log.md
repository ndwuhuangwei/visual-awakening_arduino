# 2021-7-16
## 发生了OOM但是并不是batch_size的问题
### 现象
训练resnet-50, 在3090上batch_size都下降为256了，还是OOM

提示 错误可能来源于 train-step 中的 labels 输入，指向了gradient-tape中的binary-crossentropy计算

### 原因
怀疑是 labels的格式与二进制交叉熵不配对

但最后发现就是batch_size的问题


### 解决方法
batch_size降为64


# 2021-7-17
## val set 正确率震荡
### 原因
可能是batch_size太小，也可能是过拟合

   
## WARNING:tensorflow:AutoGraph could not transform ...
### 原因
量化训练使用不正确

### 解决
因为官方教程使用model.fit，因此转用这种训练方法

## TypeError: binary_crossentropy() missing 2 required positional arguments: 'y_true' and 'y_pred'
### 原因
API使用错误，LOSS API有binary_crossentropy()和BinaryCrossentropy()两种写法

第一种需要有两个参数

### 解决方法
换一种API

## 无法存储最好的权重，WARNING:tensorflow:Can save best model only with val_accuracy available, skipping.
### 原因
model.compile中的metircs有问题
### 解决
metrics=['accuracy']

# 2021-7-18
## snowflake连接ssh显示权限问题
### 原因 
本地主机ssh设置有问题

### 解决
命令行 

ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no

## 在服务器上跑的时候 dataset出现ValueError
### 原因
数据集路径写错了

### 解决
把路径改对就好

## 训练普通模型没事，训练量化感知模型OOM
### 原因
同一个程序中先普通再量化，显存是两倍
### 解决
暂无

# 2021-7-19
## ValueError: logits and labels must have the same shape ((None, 2) vs (None, 1))
### 原因 
Image generator的 class mode选错了，binary只会生成一维的标签，只有 'categorical'才会生成二维的
### 解决
改成 'categorical'

# 2021-7-25
## after adding SEblock, there is dim error, error log shows that we can't pass 2 dim tensor to conv
### reason

In SEblock, we put the reshape in a wrong place, after global average pooling, the tensor shape will switch from 4 dim to 2 dim

### solution
put the reshape after the global average pooling

##  ValueError: tf.function-decorated function tried to create variables on non-first call.
### reason
can't directly use official layers api in call()

### solution
use self.XXX to use them indirectly

# 2021-8-4
## 下拉框建立slot机制时 TypeError: native Qt signal is not callable

