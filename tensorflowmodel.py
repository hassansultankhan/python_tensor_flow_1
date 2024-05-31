import os
import tensorflow as tf
import numpy as np

# Generate some example data
x_train = np.array([1, 2, 3, 4], dtype=np.float32)
y_train = np.array([2, 4, 6, 8], dtype=np.float32)

# Define a simple linear model
model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])

# Compile the model
model.compile(optimizer='sgd', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, epochs=500)

# Save the model
model.save('linear_model')

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_saved_model('linear_model')
tflite_model = converter.convert()

# Specify the directory to save the .tflite file
output_dir = r'C:\Users\dell\projects\tensor_flow1\assets'
os.makedirs(output_dir, exist_ok=True)

# Save the model as a .tflite file
tflite_model_path = os.path.join(output_dir, 'linear_model.tflite')
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f'Model saved to {tflite_model_path}')
