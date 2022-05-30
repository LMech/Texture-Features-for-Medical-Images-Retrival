import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog

img = cv2.imread("dataset/1/3133182.jpg")

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

fd, hogimage = hog(
    img,
    orientations=8,
    pixels_per_cell=(16, 16),
    cells_per_block=(1, 1),
    visualize=True,
)
cv2.imshow("img1", hogimage)

_, hogimage = hog(
    img,
    orientations=16,
    pixels_per_cell=(16, 16),
    cells_per_block=(1, 1),
    visualize=True,
)
cv2.imshow("img2", hogimage)

fd, hogimage = hog(
    img,
    orientations=8,
    pixels_per_cell=(32, 32),
    cells_per_block=(1, 1),
    visualize=True,
)
cv2.imshow("img3", hogimage)

fd, hogimage = hog(
    img,
    orientations=16,
    pixels_per_cell=(32, 32),
    cells_per_block=(1, 1),
    visualize=True,
)


cv2.imshow("img4", hogimage)
cv2.waitKey(0)
cv2.destroyAllWindows()
