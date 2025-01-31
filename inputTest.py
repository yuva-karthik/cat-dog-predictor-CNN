from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the saved model
model = load_model("image_classification_model.h5")

# Load and preprocess the image
img_path = r"C:\Users\satellite\OneDrive\Desktop\image detection\download.jpg"
img = image.load_img(img_path, target_size=(150, 150))  # Adjust size based on your model
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Make a prediction
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)

class_names = ['cat', 'dog']  # Replace with your actual class names

# Print with safe encoding
print(f"Predicted Class: {class_names[predicted_class]}".encode('utf-8', errors='ignore').decode())
