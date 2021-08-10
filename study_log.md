# single crop test

[深度学习中single crop / multiple crops evaluation/test 是什么意思](https://blog.csdn.net/IT_flying625/article/details/104900050)

是一种将validation set 变成 test set 的方法

将测试图像resize到某个尺度（例如256xN），选择其中center crop（即图像正中间区域，比如224x224），作为CNN的输入，去评估该模型