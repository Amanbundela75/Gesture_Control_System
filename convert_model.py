import tensorflow as tf
# Naye model ka naam yahan daalein
model = tf.keras.models.load_model('simple_gesture_model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
print("Model converted and saved as 'model.tflite'.")