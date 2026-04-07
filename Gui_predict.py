import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

# Load the trained model and class indices from the files
model = tf.keras.models.load_model('CNN_trained_model.h5')
with open('CNN_trained_model_class_indices.pkl', 'rb') as file:
    class_indices = pickle.load(file)

# Define the class labels
class_labels = list(class_indices.keys())

# Function to load and predict from an image
def load_image():
    # Allow user to select an image file
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            # Open the selected image file
            img = Image.open(file_path)
            img = img.resize((100, 100))  # Resize image to match the input size of the model
            img = ImageTk.PhotoImage(img)
            image_label.config(image=img)
            image_label.image = img
            
            # Preprocess the image for prediction
            img = image.load_img(file_path, target_size=(100, 100))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Normalize pixel values
            
            # Make prediction
            prediction = model.predict(img_array)
            predicted_class = class_labels[np.argmax(prediction)]
            result_label.config(text=f"Predicted Class: {predicted_class}")
        except Exception as e:
            result_label.config(text="Error: " + str(e))

# Create the main window
root = tk.Tk()
root.title("Image Classification")

# Create a frame for image display
image_frame = tk.Frame(root)
image_frame.pack(pady=10)

# Create a label for displaying the image
image_label = tk.Label(image_frame)
image_label.pack()

# Create a button for loading an image
load_button = tk.Button(root, text="Load Image", command=load_image)
load_button.pack(pady=10)

# Create a label for displaying the predicted result
result_label = tk.Label(root, text="")
result_label.pack(pady=10)

# Run the main event loop
root.mainloop()

