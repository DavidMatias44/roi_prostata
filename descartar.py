import cv2
import os
import pydicom
import numpy as np
import shutil


images_path = 'paciente000/'


def main():
    images = read_dicom_images_and_convert_to_array(images_path)
    filtered_images = [i for i in images if np.mean(i['data']) < 280]

    result_path = "result0/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    for filtered_image in filtered_images:
        src = filtered_image['path']
        img = str.split(filtered_image['path'], '/')[1]
        des = os.path.join(result_path, img)
        shutil.copy(src, des)
    print("Se copiaron las imagenes DICOM al directorio de resultados.")


def read_dicom_images_and_convert_to_array(images_path): 
    images = os.listdir(images_path)

    result = []
    for image in images:
        path = os.path.join(images_path, image)
        result.append({'path': path, 'data': pydicom.dcmread(path).pixel_array})
    
    return result


if __name__ == "__main__":
    main()