from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel

'''
https://build-system.fman.io/pyqt5-tutorial
'''

# The brackets [] in the above line represent the command line arguments passed to the application.
# Because our app doesn't use any parameters, we leave the brackets empty.
app = QApplication([])
window = QWidget()
layout = QVBoxLayout()
layout.addWidget(QPushButton('Top'))
layout.addWidget(QPushButton('Bottom'))
window.setLayout(layout)
window.show()
app.exec()

