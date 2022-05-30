import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
import numpy as np

img = cv2.imread("tutorials_code/DSS/lab3/zebra.jpg")

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

fd, hogimage = hog(
    img, orientations=0, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=True
)

fd, hogimage = hog(
    hogimage,
    orientations=int(np.pi / 2 * 360),
    pixels_per_cell=(8, 8),
    cells_per_block=(1, 1),
    visualize=True,
)


cv2.imshow("img", hogimage)
cv2.waitKey(0)
cv2.destroyAllWindows()
