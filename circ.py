def detectar_circulos(image):
    # Convertir a escala de grises (si no lo está ya)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplicar desenfoque para reducir ruido
    blurred = cv2.GaussianBlur(image, (15, 15), 0)

    # Detectar círculos usando la transformada de Hough
    circles = cv2.HoughCircles(blurred, 
                               cv2.HOUGH_GRADIENT, dp=1, minDist=30, 
                               param1=50, param2=30, minRadius=10, maxRadius=100)
    
    if circles is not None:
        # Convertir las coordenadas de los círculos a enteros
        circles = np.round(circles[0, :]).astype("int")

        # Dibujar los círculos sobre la imagen
        for (x, y, r) in circles:
            cv2.circle(image, (x, y), r, (0, 255, 0), 4)  # Círculo verde
            cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)  # Centro del círculo

        return image, circles
    else:
        return image, None

# Función para calcular el centroide (promedio) de los círculos detectados
def calcular_centroide(circles, dist_threshold=30):
    if circles is not None and len(circles) > 0:
        # Filtrar círculos cercanos
        nearby_circles = []
        for i, (x1, y1, _) in enumerate(circles):
            for j, (x2, y2, _) in enumerate(circles):
                if i != j:
                    dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    if dist < dist_threshold:  # Solo considerar círculos que estén cerca unos de otros
                        nearby_circles.append((x1, y1))
        
        if len(nearby_circles) > 0:
            # Calcular el promedio de las coordenadas de los círculos cercanos
            nearby_circles = np.array(nearby_circles)
            centroide_x = int(np.mean(nearby_circles[:, 0]))
            centroide_y = int(np.mean(nearby_circles[:, 1]))

            return centroide_x, centroide_y
    return None

# Cargar imagen (asegúrate de que la ruta sea correcta)
# image = cv2.imread("/Users/marti/Documents/pros2.png")
# image = cv2.imread("pruebas/resultados/roi_1-01.dcm.png")
image = cv2.imread("pruebas/resultados/roi_1-17.dcm.png")
# images = os.listDir

# Detectar círculos
image_with_circles, detected_circles = detectar_circulos(image)

# Si se detectaron círculos, calcular el centroide
if detected_circles is not None:
    centroide = calcular_centroide(detected_circles, dist_threshold=35)  # Distancia mínima entre círculos
    if centroide:
        centroide_x, centroide_y = centroide
        # Dibujar el centroide (punto pequeño)
        cv2.circle(image_with_circles, (centroide_x, centroide_y), 5, (0, 0, 255), -1)  # Centroide más pequeño en rojo

# Mostrar imagen con círculos detectados y el centroide
plt.imshow(cv2.cvtColor(image_with_circles, cv2.COLOR_BGR2RGB))
plt.title("Círculos y Centroide Detectado")
plt.axis("off")
plt.show()

# Mostrar coordenadas y radios de los círculos detectados
if detected_circles is not None:
    print("Círculos detectados:")
    for (x, y, r) in detected_circles:
        print(f"Centro: ({x}, {y}), Radio: {r}")
    print(f"Centroide de los círculos: ({centroide_x}, {centroide_y})")
else:
    print("No se detectaron círculos.")