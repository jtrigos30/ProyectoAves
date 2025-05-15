import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# --- Cargar el modelo Keras (.keras) ---
@st.cache_resource
def cargar_modelo():
    modelo = tf.keras.models.load_model("model_VGG16_v1.keras")
    return modelo

# --- Preprocesamiento para VGG16 ---
def preparar_imagen_vgg16(imagen):
    imagen = imagen.convert("RGB")  # Asegurar que tenga 3 canales
    imagen = imagen.resize((224, 224))  # Tamaño requerido por VGG16
    matriz = np.array(imagen).astype(np.float32) / 255.0  # Normalizar
    matriz = np.expand_dims(matriz, axis=0)  # Añadir dimensión de batch
    return matriz

# --- Etiquetas del modelo ---
etiquetas = [
    'CHIRIGUE AZAFRANADO', 'CHIRIGUE CITRINO', 'COLUDO COLICUNA', 'GORRION CANARIO SABANERO', 
    'HEMISPINGO CABECINEGRO', 'SEMILLERO BRINCADOR', 'SEMILLERO PECHO CANELA', 'SEMILLERO PIZARRA', 
    'TANGARA PIQUIRRUBIO', 'YAL PLOMIZO'

]

# --- Título de la aplicación ---
st.title("🦜 Clasificación de Aves con VGG16 + Keras")
st.write("Sube una imagen de un ave para predecir su especie.")

# --- Cargar imagen del usuario ---
archivo_imagen = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])

if archivo_imagen:
    imagen = Image.open(archivo_imagen)
    st.image(imagen, caption="Imagen cargada", use_container_width=True)

    imagen_preparada = preparar_imagen_vgg16(imagen)

    # --- Cargar modelo e inferencia ---
    modelo = cargar_modelo()
    salida_predicha = modelo.predict(imagen_preparada)

    clase = int(np.argmax(salida_predicha))
    confianza = float(np.max(salida_predicha))

    st.success(f"🧠 Predicción: *{etiquetas[clase]}*")
    st.info(f"📊 Confianza del modelo: *{confianza*100:.2f}%*")

    # --- Visualización opcional ---
    if st.checkbox("Mostrar probabilidades por clase"):
        st.bar_chart(salida_predicha[0])