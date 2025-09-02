import tensorflow as tf

# Apne Keras model (.h5 file) ko load karein
model = tf.keras.models.load_model('hand_gesture_model.h5')
print("H5 model loaded successfully.")

# TFLite converter banayein
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Conversion process run karein
tflite_model = converter.convert()
print("Model converted to TensorFlow Lite format.")

# Naye .tflite model ko save karein
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("New model saved as 'model.tflite'. ✅")