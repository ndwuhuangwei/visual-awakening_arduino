import sys
import os
import cv2
import time

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QPalette, QBrush, QPixmap
from predict import predict_img, load_model, face_detect

from PIL import Image

# model = None

global weights_dir


class Ui_MainWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)
        
        self.timer_camera = QtCore.QTimer()  # 初始化定时器
        self.cap = cv2.VideoCapture()  # 初始化摄像头
        self.CAM_NUM = 0
        self.set_ui()
        self.slot_init()
        self.__flag_work = 0
        self.x = 0
        self.count = 0
        
        # self.model = ''
        # self.cost_time = 0.0
        # self.model_size = 0.0
        # self.model_params = 0.0
        # self.weights_dir = ''

        self.button_open_camera = QtWidgets.QPushButton(u'打开相机')
        self.button_close = QtWidgets.QPushButton(u'退出')
    
    def set_ui(self):
        # 定义一些全局变量
        self.model = ''
        self.cost_time = 0.0
        self.model_size = 0.0
        self.model_params = 0.0
        self.weights_dir = ''
        self.isCameraOpen = False
        self.modelType = 'MicroNet'
        
        # 设置空间摆放顺序，不重要
        self.__layout_main = QtWidgets.QHBoxLayout()  # 采用QHBoxLayout类，按照从左到右的顺序来添加控件
        self.__layout_fun_button = QtWidgets.QHBoxLayout()
        self.__layout_data_show = QtWidgets.QVBoxLayout()  # QVBoxLayout类垂直地摆放小部件

        ###################################################################
        # 按钮设置
        ###################################################################
        
        self.button_open_camera = QtWidgets.QPushButton(u'开始检测')
        self.button_close = QtWidgets.QPushButton(u'退出')
        self.button_chooseCkpt = QtWidgets.QPushButton(u'选择权重')
        self.button_chooseModel = QtWidgets.QComboBox(self)

        infomation = ["MicroNet", "MobileNetV2", "MobileNetV2_Inv"]
        self.button_chooseModel.addItems(infomation)

        # self.button_chooseModel.activated(str).connect(self.choose_model)
        
        # button颜色修改
        # 按钮样式设置，不重要
        button_color = [self.button_open_camera, self.button_close]
        '''
        background-color: 按钮背景色
        '''
        for i in range(2):
            button_color[i].setStyleSheet("QPushButton{color:black}"
                                          "QPushButton:hover{color:red}"
                                          "QPushButton{background-color:rgb(255,255,255)}"
                                          "QpushButton{border:2px}"
                                          "QPushButton{border_radius:10px}"
                                          "QPushButton{padding:2px 4px}")
        
        self.button_open_camera.setMinimumHeight(80)
        self.button_open_camera.setMinimumWidth(120)
        self.button_close.setMinimumHeight(80)
        self.button_close.setMinimumWidth(120)
        self.button_chooseModel.setMinimumHeight(80)
        self.button_chooseModel.setMinimumWidth(120)
        self.button_chooseCkpt.setMinimumHeight(80)
        self.button_chooseCkpt.setMinimumWidth(120)
        
        # move()方法是移动窗口在屏幕上的位置到x = 500，y = 500的位置上
        self.move(500, 500)
        
        ###################################################################
        # 信息显示 label
        ###################################################################
        
        self.label_show_camera = QtWidgets.QLabel()
        self.label_move = QtWidgets.QLabel()
        
        self.show_costTime = QtWidgets.QLabel()
        self.show_paraNum = QtWidgets.QLabel()
        self.show_memSize = QtWidgets.QLabel()
        
        self.label_move.setFixedSize(100, 100)
        
        self.label_show_camera.setFixedSize(641, 481)
        self.label_show_camera.setAutoFillBackground(True)
        
        self.__layout_fun_button.addWidget(self.button_open_camera)
        self.__layout_fun_button.addWidget(self.button_close)
        self.__layout_fun_button.addWidget(self.label_move)
        self.__layout_fun_button.addWidget(self.button_chooseModel)
        self.__layout_fun_button.addWidget(self.button_chooseCkpt)
        
        self.__layout_fun_button.addWidget(self.show_costTime)
        self.__layout_fun_button.addWidget(self.show_paraNum)
        self.__layout_fun_button.addWidget(self.show_memSize)
        
        self.__layout_main.addLayout(self.__layout_fun_button)
        self.__layout_main.addWidget(self.label_show_camera)
        
        self.setLayout(self.__layout_main)
        self.label_move.raise_()
        
        self.setWindowTitle(u'人员检测')

        # 显示一些模型信息
        # self.show_costTime.setText(u'  模型加载时间: {0:.2f} s'.format(self.cost_time))
        # self.show_paraNum.setText(u'  模型参数量: {0:.2f} Million'.format(self.model_params/1000000))
        # self.show_memSize.setText(u'  模型尺寸: {} MB'.format(self.model_size))
        self.show_analysis()
        
        '''
        # 设置背景颜色
        palette1 = QPalette()
        palette1.setBrush(self.backgroundRole(),QBrush(QPixmap('background.jpg')))
        self.setPalette(palette1)
        '''
    
    def slot_init(self):  # 建立通信连接
        self.button_open_camera.clicked.connect(self.button_open_camera_click)
        self.timer_camera.timeout.connect(self.show_camera)
        
        self.button_close.clicked.connect(self.close)
        self.button_chooseCkpt.clicked.connect(self.choose_ckpt)
        self.button_chooseModel.activated.connect(self.choose_model)
    
    def button_open_camera_click(self):
        if self.timer_camera.isActive() == False:
            flag = self.cap.open(self.CAM_NUM)
            if flag == False:
                msg = QtWidgets.QMessageBox.Warning(self, u'Warning', u'请检测相机与电脑是否连接正确',
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
                # if msg==QtGui.QMessageBox.Cancel:
                #                     pass
            else:
                self.timer_camera.start(30)
                self.button_open_camera.setText(u'开始检测')
                self.isCameraOpen = True
        else:
            self.timer_camera.stop()
            self.cap.release()
            self.label_show_camera.clear()
            self.button_open_camera.setText(u'结束检测')
            self.isCameraOpen = False
            
    # 放置模型的区域
    # 打开相机的时候，这个函数内的代码就会被执行
    def show_camera(self):
        time_predict_start = time.time()
        
        # 从摄像头读取图像
        flag, self.image = self.cap.read()
        show = cv2.resize(self.image, (640, 480))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        
        # 进行推理
        cap_img = Image.fromarray(show)
        result_class, result_prob = predict_img(img=cap_img, my_model=self.model)
        
        # 计算fps
        time_predict_end = time.time()
        fps = int(1/(time_predict_end - time_predict_start))
        
        # 人脸检测
        x, y, w, h = face_detect(show)
        if x != 0 and y != 0 and w != 0 and h != 0:
            cv2.rectangle(show, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        # 显示图像
        result_class_text = "Result: " + result_class
        result_prob_text = "Prob: {0:.4f}".format(result_prob)
        fps_text = "FPS: {}".format(fps)
        cv2.putText(show, result_class_text, (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (200, 255, 255), 1)
        cv2.putText(show, result_prob_text, (280, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (200, 255, 255), 1)
        cv2.putText(show, fps_text, (480, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (200, 255, 255), 1)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))
    
    def choose_ckpt(self):
        self.button_chooseCkpt.setText(u'正在加载权重...')
        if self.isCameraOpen == False:
            self.weights_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "getExistingDirectory", "./")
            if self.weights_dir != None:
                self.model, self.cost_time, self.model_size, self.model_params = \
                    load_model(weights_dir=self.weights_dir, model_type=self.modelType)
                self.button_chooseCkpt.setText(u'选择权重')
                self.show_analysis()

    def closeEvent(self, event):
        ok = QtWidgets.QPushButton()
        cancel = QtWidgets.QPushButton()
        msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, u'关闭', u'是否关闭！')
        msg.addButton(ok, QtWidgets.QMessageBox.ActionRole)
        msg.addButton(cancel, QtWidgets.QMessageBox.RejectRole)
        ok.setText(u'确定')
        cancel.setText(u'取消')
        if msg.exec_() == QtWidgets.QMessageBox.RejectRole:
            event.ignore()
        else:
            if self.cap.isOpened():
                self.cap.release()
            if self.timer_camera.isActive():
                self.timer_camera.stop()
            event.accept()
            
    def show_analysis(self):
        # 换权重后显示模型参数
        self.show_costTime.setText(u'  模型加载时间: {0:.2f} s'.format(self.cost_time))
        self.show_paraNum.setText(u'  模型参数量: {0:.2f} Million'.format(self.model_params / 1000000))
        self.show_memSize.setText(u'  模型尺寸: {} MB'.format(self.model_size))
    
    def choose_model(self):
        self.modelType = self.button_chooseModel.currentText()
        

if __name__ == '__main__':
    App = QApplication(sys.argv)
    win = Ui_MainWindow()
    win.show()
    sys.exit(App.exec_())
