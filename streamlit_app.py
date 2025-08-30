import streamlit as st
import pickle
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

# Define the preprocessing function here (same as in your notebook)
def preprocess_data(X):
    X = X.astype('float32') / 255.0
    return X

# Load only the model (use 'model.pkl' which contains just the model)
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# App title and description
st.title('MNIST Digit Classifier')
st.write("""
This app uses a neural network trained on the MNIST dataset to predict handwritten digits.
Upload a grayscale image of a digit (preferably 28x28 pixels) for prediction.
""")

# File uploader
uploaded_file = st.file_uploader("Upload a digit image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # Open and process the image
        image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
        image = image.resize((28, 28))  # Resize to model input shape
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Convert to numpy array and preprocess
        image_array = np.array(image)
        processed_image = preprocess_data(image_array)
        processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension (1, 28, 28)
        
        # Make prediction
        prediction = model.predict(processed_image)
        predicted_digit = np.argmax(prediction[0])
        confidence = np.max(prediction[0]) * 100  # Confidence percentage
        
        # Display results
        st.subheader('Prediction')
        st.write(f"Predicted Digit: **{predicted_digit}**")
        st.write(f"Confidence: **{confidence:.2f}%**")
        
        # Optional: Show probability distribution
        if st.checkbox("Show probability distribution"):
            st.bar_chart(prediction[0])
    
    except Exception as e:
        st.error(f"Error processing image: {str(e)}. Ensure it's a valid image file.")

# Additional info section
st.sidebar.title('Model Details')
st.sidebar.write("""
- **Architecture**: Flatten -> Dense(128, ReLU) -> Dense(32, ReLU) -> Dense(10, Softmax)
- **Training**: 15 epochs on MNIST, ~97.5% validation accuracy (from notebook)
- **Input**: 28x28 grayscale images, scaled to [0,1]
""")