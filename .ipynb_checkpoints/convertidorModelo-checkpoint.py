# El siguiente script convierte el modelo a .keras para ser leido y ejecutado en el stramlit
import tensorflow as tf

modelo = tf.keras.models.load_model(r"C:\Users\juant\DataSet Aves\Modelo entrenado") 


converter = tf.lite.TFLiteConverter.from_keras_model(modelo)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, 
    tf.lite.OpsSet.SELECT_TF_OPS
]
tflite_model = converter.convert()
# Guardarlo como archivo .tflite
with open("modeloPresentacionJd.tflite", "wb") as f:
    f.write(tflite_model)
    