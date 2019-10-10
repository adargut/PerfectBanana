"""
Convert a TensorFlow model to TF Lite model to be used on Android
"""
import tensorflow as tf

# Load existing model.
model = tf.keras.models.load_model('output/keras_vggnet.model')
model.summary()

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Write converted model.
open("converted_model.tflite", "wb").write(tflite_model)
