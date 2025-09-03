import tensorflow as tf

# Naye model (.h5 file) ka naam yahan daalein
model_name = 'hand_gesture_model_asl.h5'

model = tf.keras.models.load_model(model_name)
print(f"'{model_name}' loaded successfully.")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
print("Model converted to TensorFlow Lite format.")

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
print("New model saved as 'model.tflite'. ✅")