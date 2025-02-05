import os
import pydicom
import cv2
import numpy as np
import matplotlib.pyplot as plt


def detectar_circulos(image, image_half, threshold_x, threshold_y):
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

        return image, circles
    else:
        return image, None


def read_dicom_images(images_path):
    images = os.listdir(images_path)

    result = []
    for image in images:
        path = os.path.join(images_path, image)
        temp = pydicom.dcmread(path).pixel_array
        temp = cv2.normalize(temp, None, 0, 255, cv2.NORM_MINMAX)
        temp = np.uint8(temp)

        result.append(temp)

    return result


def calcular_centroide(circles, dist_threshold=30):
    if circles is not None and len(circles) == 1:
        (x, y, _) = circles[0]
        return x, y
    elif circles is not None and len(circles) > 0:
        nearby_circles = []
        for i, (x1, y1, _) in enumerate(circles):
            for j, (x2, y2, _) in enumerate(circles):
                if i != j:
                    dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    if dist < dist_threshold:
                        nearby_circles.append((x1, y1))
        
        if len(nearby_circles) > 0:
            nearby_circles = np.array(nearby_circles)
            centroide_x = int(np.mean(nearby_circles[:, 0]))
            centroide_y = int(np.mean(nearby_circles[:, 1]))

            return centroide_x, centroide_y
    return None


def asdf(images):
    result = []
    seeds = []

    for image in images:
        width, height = image.shape
        image_half = width // 2 

        image_original = np.copy(image)
        image_with_circles, detected_circles = detectar_circulos(image, image_half=image_half, threshold_x=width // 10, threshold_y=height // 10)

        if detected_circles is not None:
            centroide = calcular_centroide(detected_circles, dist_threshold=100)  # Distancia mínima entre círculos
            if centroide:
                centroide_x, centroide_y = centroide
                seeds.append((centroide_y, centroide_x))
                cv2.circle(image_with_circles, (centroide_x, centroide_y), 5, (0, 0, 255), -1)  # Centroide más pequeño en rojo

        if detected_circles is not None:
            print("Círculos detectados:")
            for (x, y, r) in detected_circles:
                print(f"Centro: ({x}, {y}), Radio: {r}")
            print(f"Centroide de los círculos: ({centroide_x}, {centroide_y})\n")
        else:
            print("No se detectaron círculos.\n")


        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Imagen original")
        plt.imshow(image_original)
        plt.subplot(1, 2, 2)
        plt.title("Detección de círculos")
        plt.imshow(image)

        plt.show()

def main():
    images_path = 'prueba_temo/'

    images = read_dicom_images(images_path) 

    images = asdf(images)


main()