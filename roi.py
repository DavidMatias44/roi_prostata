import pydicom
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology


images_path = "result/"


def read_dicom_images(images_path):
    images = os.listdir(images_path)

    result = []
    for image in images:
        path = os.path.join(images_path, image)
        result.append(pydicom.dcmread(path).pixel_array)

    return result


def region_growing_multiple_seeds(image, seeds, threshold):
    height, width = image.shape
    visited = np.zeros_like(image, dtype=bool)
    region = np.zeros_like(image, dtype=np.uint8)

    # Inicializar la cola de puntos con todas las semillas
    stack = seeds[:]

    # Procesar cada punto en la cola
    while stack:
        y, x = stack.pop()
        if visited[y, x]:
            continue

        visited[y, x] = True

        # Verificar si el píxel pertenece a la región
        for seed_y, seed_x in seeds:
            if abs(int(image[y, x]) - int(image[seed_y, seed_x])) <= threshold:
                region[y, x] = 255

                # Expandir a los vecinos
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and not visited[ny, nx]:
                        stack.append((ny, nx))

    return region
    

def detectar_circulos(image):
    blurred = cv2.GaussianBlur(image, (15, 15), 0)

    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, dp=1, minDist=15, 
        param1=30, param2=25, minRadius=5, maxRadius=100
    )
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        for (x, y, r) in circles:
            cv2.circle(image, (x, y), r, (0, 255, 0), 4)  # Círculo verde
            cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)  # Centro del círculo
        return image, circles
    else:
        return image, None


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


def main():
    images = read_dicom_images(images_path)
    seeds = []

    for image in images:
        image_height, image_width = image.shape

        image_center = (image_width // 2, image_height // 2) 
        roi_width = image_width // 3
        roi_height = image_height // 3

        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)  # Normaliza valores a 0-255
        image = np.uint8(image)

        roi_start = (image_center[0] - roi_width // 2, image_center[1] - roi_height // 2)
        roi = image[roi_start[1]:roi_start[1] + roi_height, roi_start[0]:roi_start[0] + roi_width]
        roi_original = np.copy(roi)

        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # contrasted = clahe.apply(roi)
        # contrasted_original = np.copy(contrasted)

        image_with_circles, detected_circles = detectar_circulos(roi)

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

        # image_eq = cv2.equalizeHist(roi_original.astype(np.uint8))  # Mejora del contraste (asegurando que es uint8)
        image_blur = cv2.GaussianBlur(roi_original, (5, 5), 0)  # Suavizado

        region = region_growing_multiple_seeds(image_blur, seeds, threshold=4)

        dark_mask = region < 90 # Excluir píxeles oscuros (ajusta según sea necesario)
        region_filtered = image_blur * dark_mask

        # region_cleaned = morphology.remove_small_objects(region_filtered.astype(bool), min_size=500).astype(np.uint8) * 255

        # kernel = np.ones((5, 5), np.uint8)
        # region_cleaned = cv2.morphologyEx(region_cleaned, cv2.MORPH_OPEN, kernel)  # Eliminar conexiones delgadas


        # Visualizar resultados
        # plt.figure(figsize=(12, 6))
        # plt.subplot(1, 2, 1)
        # plt.title("Imagen original")
        # plt.imshow(roi_original, cmap='gray')
        # plt.subplot(1, 2, 2)
        # plt.title("Detección de círculos")
        # plt.imshow(roi, cmap='gray')
        plt.subplot(1, 2, 1)
        plt.title("Crecimiento de regiones")
        plt.imshow(region, cmap='gray')
        plt.subplot(1, 2, 2)
        plt.title("Crecimiento de regiones procesada")
        plt.imshow(region_filtered, cmap='gray')

        plt.show()

main()