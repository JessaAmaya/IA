import cv2 as cv
import numpy as np

# Cargar el clasificador en cascada para detectar caras
wally_cascade = cv.CascadeClassifier(r'D:\Escuela\Ultimooo\IA\Buscando_a_Wally\cascade.xml')

# Cargar la imagen en la que deseas buscar a Wally
img = cv.imread(r'D:\Escuela\Ultimooo\IA\Buscando_a_Wally\fondos\15.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Detectar caras en la imagen
wally = wally_cascade.detectMultiScale(gray, scaleFactor=1.01,minNeighbors=15, minSize=(30, 30))

# Dibujar un rectángulo alrededor de cada cara detectada
for (x, y, w, h) in wally:
    
    # Calcular el porcentaje de coincidencia
    porcentaje_coincidencia = ((w * h) / (img.shape[0] * img.shape[1])) * 100
    # Dibujar el rectángulo
    cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Mostrar la imagen con los rectángulos dibujados
cv.namedWindow('Imagen con Wally detectado', cv.WINDOW_NORMAL)
cv.resizeWindow('Imagen con Wally detectado', 1080, 720)
cv.imshow('Imagen con Wally detectado', img)

cv.waitKey(0)
cv.destroyAllWindows()
