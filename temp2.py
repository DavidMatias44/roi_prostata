import os
import pydicom
import cv2
import numpy as np
import matplotlib.pyplot as plt


def read_dicom_images(path):
    images = os.listdir(path)

    result = []
    for image in images:
        path_ = os.path.join(path, image)
        
        temp = pydicom.dcmread(path_).pixel_array
        temp = cv2.normalize(temp, None, 0, 255, cv2.NORM_MINMAX)

        result.append(np.uint8(temp))

    return result


def get_roi_coords(image):
    width, height = image.shape
    image_half = width // 2
    
    threshold_x = width // 10
    threshold_y = height // 10

    blurred = cv2.GaussianBlur(image, (15, 15), 0)

    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, dp=1, minDist=15, 
        param1=30, param2=30, minRadius=5, maxRadius=100
    )
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        filtered_circles = []

        for (x, y, r) in circles:
            if ((image_half + threshold_x > x) and (image_half - threshold_x < x) and
               (image_half + threshold_y > y) and (image_half - threshold_y < y)):
                cv2.circle(image, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
                filtered_circles.append((x, y, r))

        x_min = min([x - r for x, y, r in filtered_circles])
        y_min = min([y - r for x, y, r in filtered_circles])
        x_max = max([x + r for x, y, r in filtered_circles])
        y_max = max([y + r for x, y, r in filtered_circles])

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # SI SE QUIERE VER EL RESULTADO
        # plt.figure(figsize=(12, 6))
        # plt.subplot(1, 1, 1)
        # plt.title("Imagen con ROI detectado.")
        # plt.imshow(image)
        # plt.show()

        return (x_min, y_min, x_max, y_max)
    else:
        return None 


def main():
    images_path = 'result/'
    images = read_dicom_images(images_path)

    roi_coords = []
    for image in images:
        # el primer par de coordenadas es el punto de mas arriba a la izquierda.
        # el segundo par de coordenadas es el punto de mas abajo a la derecha.
        roi_coords.append(get_roi_coords(image)) 

    print(roi_coords)


main()