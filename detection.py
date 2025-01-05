import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw
import os

# Load your trained model
try:
    model = load_model('model/model_v3/model1.h5')  # Replace with your actual path
    print("Model Loaded Successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Class Names
class_names = ["electric bus", "electric car"]  # Adjust based on the order of your classes

# Function to preprocess the image for your model
IMG_SIZE = 150  # Match the size used in your model training


def preprocess_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')  # Load with PIL and convert to RGB
        resized_image = image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)  # Resize with PIL using correct resampling
        image_array = np.array(resized_image)  # Convert to numpy array
        image_array = np.expand_dims(image_array, axis=0)
        image_array = image_array / 255.0  # Rescale to the range of 0-1
        return image_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None


def process_image(image_path):
    try:
        processed_image = preprocess_image(image_path)
        if processed_image is None:
            print(f"Failed to pre-process {image_path}")
            return

        if model:
            # Make a prediction
            prediction = model.predict(processed_image)
            predicted_class_index = np.argmax(prediction)
            predicted_class_name = class_names[predicted_class_index]
            print(f"Predicted class: {predicted_class_name}")
        else:
            print("Model not loaded. Cannot make a prediction")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    image_path = "datasets/test/training/electric_bus/images-125.jpeg"  # Replace with your actual image path
    if os.path.exists(image_path):
        process_image(image_path)
    else:
        print(f"Path: {image_path} does not exist")