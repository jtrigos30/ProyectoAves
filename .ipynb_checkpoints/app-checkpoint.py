import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# --- Cargar el modelo TFLite ---
@st.cache_resource
def cargar_modelo():
    interpreter = tf.lite.Interpreter(model_path="model_vgg16_val_accuracy_0.6602.tflite") # Aqui se coloca el modelo teniendo en cuenta la ruta y el nombre y la extension
    interpreter.allocate_tensors()
    return interpreter

# --- Preprocesamiento para VGG16 ---
def preparar_imagen_vgg16(imagen):
    imagen = imagen.convert("RGB")  # Asegurar que tenga 3 canales
    imagen = imagen.resize((224, 224))  # Tamaño requerido por VGG16
    matriz = np.array(imagen).astype(np.float32) / 255.0  # Normalizar
    matriz = np.expand_dims(matriz, axis=0)  # Añadir dimensión de batch
    return matriz

# --- Etiquetas del modelo ---
etiquetas = [
    'CHRYOSOMUS ICTEROCEPHALUS', 'GYMNOMYSTAX MEXICANUS', 'HYPOPYRRHUS PYROHYPOGASTER', 'PARKESIA MOTACILLA', 
    'PARKESIA MOTACILLA_NOVEBORACENSIS', 'PARKESIA NOVEBORACENSIS', 'QUISCALUS LUGUBRIS', 'QUISCALUS MAJOR', 
    'QUISCALUS MEXICANUS', 'VERMIVORA CHRYSOPTERA'
]

# --- Título de la aplicación ---
st.title("🦜 Clasificación de Aves con VGG16 + TensorFlow Lite")
st.write("Sube una imagen de un ave para predecir su especie.")

# --- Cargar imagen del usuario ---
archivo_imagen = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])

if archivo_imagen:
    imagen = Image.open(archivo_imagen)
    st.image(imagen, caption="Imagen cargada", use_container_width=True)

    imagen_preparada = preparar_imagen_vgg16(imagen)

    # --- Cargar modelo e inferencia ---
    interpreter = cargar_modelo()
    entrada = interpreter.get_input_details()
    salida = interpreter.get_output_details()

    interpreter.set_tensor(entrada[0]['index'], imagen_preparada)
    interpreter.invoke()
    salida_predicha = interpreter.get_tensor(salida[0]['index'])

    clase = int(np.argmax(salida_predicha))
    confianza = float(np.max(salida_predicha))

    st.success(f"🧠 Predicción: *{etiquetas[clase]}*")
    st.info(f"📊 Confianza del modelo: *{confianza*100:.2f}%*")

    # --- Visualización opcional ---
    if st.checkbox("Mostrar probabilidades por clase"):
        st.bar_chart(salida_predicha[0])