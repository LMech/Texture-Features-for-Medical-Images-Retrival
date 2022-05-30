import sys
from cProfile import label

from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import query_backend


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("DSS Project")

        self.layout = QVBoxLayout()
        self.layout2 = QHBoxLayout()
        self.layout2.setSpacing(0)
        self.main_img = QLabel()

        self.layout.addWidget(self.main_img)
        button = QPushButton("Press Me!")
        button.setCheckable(True)
        button.clicked.connect(self.the_button_was_clicked)

        self.layout.addWidget(button)
        self.layout.addLayout(self.layout2)

        widget = QWidget()
        widget.setLayout(self.layout)
        # Set the central widget of the Window.
        self.setCentralWidget(widget)

    def the_button_was_clicked(self):
        file_name = QFileDialog.getOpenFileName(
            self, caption="Open Image", filter=("Image Files (*.png *.jpg *.bmp)")
        )
        pixmap = QPixmap(file_name[0])
        self.main_img.setPixmap(pixmap)
        for res in query_backend.search_query(file_name[0]):
            print(res)
            main_img = QLabel()
            pixmap = QPixmap(res[1])
            main_img.setPixmap(pixmap.scaled(128, 128))
            self.layout2.addWidget(main_img)


app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()
