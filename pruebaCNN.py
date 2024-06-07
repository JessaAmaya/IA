import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import matplotlib.image as mpimg
import tensorflow as tf

# Se carga el modelo de la CNN
saved_model_path = r"D:\Escuela\Ultimooo\IA\DESASTRES\modelo"

# Se convierte a legacy
loaded = tf.saved_model.load(saved_model_path)
print(list(loaded.signatures.keys()))  # Imprime las firmas 

# Inspect the default signature
infer = loaded.signatures['serving_default']
print(infer.structured_outputs)  # Imprime la estructura del output tensor

# Establecemos las etiquetas
desastres = ['asaltos', 'incendios', 'inundaciones', 'robocasahabitacion', 'tornados']

# Funcion de procesar y predecir la imagen
def process_and_predict_image(filepath):
 image = mpimg.imread(filepath)
 image_resized = resize(image, (30, 30), anti_aliasing=True, clip=False, preserve_range=True)
 image_resized = np.expand_dims(image_resized, axis=0)  
 image_resized = image_resized.astype('float32') / 255.  # Normalizamos la imagen

 # Prepara el input tensor
 input_tensor = tf.convert_to_tensor(image_resized)
 # Hace la prediccion
 predictions = infer(input_tensor)
 output_tensor_name = list(predictions.keys())[0]  # aplicamos el output tensor de forma dinamica
 predicted_classes = predictions[output_tensor_name]  # Accedemos al output tensor
 predicted_label = desastres[tf.argmax(predicted_classes, axis=1).numpy()[0]]

 return predicted_label, image

# Cargamos la imagen a predecir
filenames = [r'D:\Escuela\Ultimooo\IA\DESASTRES\pruebas\op2.jpg']

# Procesamos cada imagen (en caso de) y las desplegamos
for filepath in filenames:
 prediction, image = process_and_predict_image(filepath)
 
 plt.imshow(image)
 plt.title(f'Yo digo que es de: {prediction}')
 plt.axis('off')
 plt.show()