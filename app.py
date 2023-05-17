import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image

# Load the saved model
model = joblib.load('trained_model.pkl')

# Define the classes
classes = {0: 'No Tumor', 1: 'Positive Tumor'}

def preprocess_image(image):
    # Preprocess the image as required (resize, normalize, etc.)
    processed_image = cv2.resize(image, (200, 200))
    processed_image = processed_image.reshape(1, -1) / 255.0
    return processed_image

def predict(image):
    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make predictions using the loaded model
    prediction = model.predict(processed_image)

    # Map the predicted class to its corresponding label
    predicted_class = classes[prediction[0]]

    return predicted_class

def main():
    # Streamlit app title and description
    st.title('Brain Tumor Classification')
    st.write('Upload an MRI brain image and get the tumor classification result.')

    # File upload section
    uploaded_file = st.file_uploader('Upload an MRI brain image', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Open the image file using PIL
        image = Image.open(uploaded_file)
        # Convert the image to grayscale
        image_gray = image.convert('L')
        # Convert the PIL image to NumPy array
        image_array = np.array(image_gray)

        # Display the uploaded image
        st.image(image_array, caption='Uploaded MRI Image', use_column_width=True)

        # Perform prediction
        result = predict(image_array)

        # Display the predicted result
        st.write('Prediction:', result)

# Run the app
if __name__ == '__main__':
    main()
