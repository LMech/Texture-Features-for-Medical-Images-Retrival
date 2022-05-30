import sys

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QLabel,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QWidget,
)

import query_backend


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.radio_buttons = ["Original", "HoG"]
        self.bottom_gallery = []
        self.setWindowTitle("My App")

        self.layout = QGridLayout()

        upload_button = QPushButton("Upload image")
        upload_button.setCheckable(True)
        upload_button.clicked.connect(self.upload_image)
        self.layout.addWidget(upload_button, 0, 0)

        search_button = QPushButton("Search")
        search_button.setCheckable(True)
        search_button.clicked.connect(self.query_search)
        self.layout.addWidget(search_button, 0, 1)

        choose_label = QLabel("Choose descriptor")
        self.layout.addWidget(choose_label, 1, 0)

        self.groupBox = QGroupBox()

        self.original_radio_button = QRadioButton(self.radio_buttons[0], self.groupBox)
        self.original_radio_button.setChecked(True)
        self.layout.addWidget(self.original_radio_button, 1, 1)

        self.hog_radio_button = QRadioButton(self.radio_buttons[1], self.groupBox)
        self.layout.addWidget(self.hog_radio_button, 1, 2)

        self.img_path_label = QLabel("No chosen images")
        self.layout.addWidget(
            self.img_path_label,
            2,
            0,
            1,
            4
        )

        self.img = QLabel()
        self.change_img(self.img)
        self.layout.addWidget(self.img, 3, 0)

        for i in range(10):
            img = QLabel()
            self.change_img(img)
            self.layout.addWidget(img, 4, i)
            sim = QLabel()
            self.layout.addWidget(sim, 5, i)
            cls = QLabel()
            self.layout.addWidget(cls, 6, i)
            self.bottom_gallery.append([img, sim, cls])

        widget = QWidget()
        widget.setLayout(self.layout)
        self.setCentralWidget(widget)

    def upload_image(self):
        file_name = QFileDialog.getOpenFileName(
            self, caption="Open Image", filter=("Image Files (*.png *.jpg *.bmp)")
        )
        self.img_path_label.setText(file_name[0])
        self.change_img(self.img, file_name[0])

    def query_search(self):
        if self.img_path_label.text() != "No chosen images":
            self.display_images(
                query_backend.search_query(
                    self.img_path_label.text(),
                    str(self.radio_buttons[self.get_descriptor()]).lower(),
                )
            )

    def change_img(self, qlabel: QLabel, img_path: str = "imgs/place_holder.png"):
        pixmap = QPixmap(img_path).scaled(128, 128)
        qlabel.setPixmap(pixmap)

    def get_descriptor(self):
        if self.original_radio_button.isChecked():
            return 0
        if self.hog_radio_button.isChecked():
            return 1

    def display_images(self, retrived_imgs):
        for i in range(10):
            img = self.bottom_gallery[i][0]
            self.change_img(img, retrived_imgs[i][1])
            sim = self.bottom_gallery[i][1]
            sim.setText(f"similarity: {int(retrived_imgs[i][0] * 100)}%")
            cls = self.bottom_gallery[i][2]
            cls.setText(f"class: {retrived_imgs[i][2]}")


app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()
