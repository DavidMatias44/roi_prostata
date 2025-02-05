import pydicom
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

images_path = "paciente001/"
# images_path = "result/"
images = os.listdir(images_path)

for image in images:
    img_path = os.path.join(images_path, image)
    img = pydicom.dcmread(img_path).pixel_array

    plt.imshow(img, cmap='gray')
    plt.show()
