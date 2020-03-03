from PyQt5.QtWidgets import *


app = QApplication([])
label = QLabel('Hello World!')
button = QPushButton('Click')

def on_button_clicked():
    alert = QMessageBox()
    alert.setText('You clicked the button!')
    alert.exec_()


button.clicked.connect(on_button_clicked)
button.show()
label.show()
app.exec_()