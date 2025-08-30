import streamlit as st
import pickle
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Define the preprocessing function here (same as in your notebook)
def preprocess_data(X):
    X = X.astype('float32') / 255.0
    return X

# Load the pipeline and extract the model
@st.cache_resource
def load_model_pipeline():
    with open('mnist_pipeline.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    return pipeline['model']

model = load_model_pipeline()

# App title and description
st.title('MNIST Digit Classifier')
st.write("""
This app uses a neural network trained on the MNIST dataset to predict handwritten digits.
You can either upload an image or draw a digit below.
""")

# Create tabs for upload and draw
tab1, tab2 = st.tabs(["Upload Image", "Draw Digit"])

with tab1:
    # File uploader
    uploaded_file = st.file_uploader("Upload a digit image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        try:
            # Open and process the image
            image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
            image = image.resize((28, 28))  # Resize to model input shape
            st.image(image, caption='Uploaded Image', use_container_width=True)
            
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
            if st.checkbox("Show probability distribution (Upload)"):
                st.bar_chart(prediction[0])
        
        except Exception as e:
            st.error(f"Error processing uploaded image: {str(e)}. Ensure it's a valid image file.")

with tab2:
    # Drawing canvas with clear button
    st.write("Draw a digit below:")
    
    # Add clear button
    col1, col2 = st.columns([3, 1])
    with col2:
        clear_canvas = st.button("🗑️ Clear Canvas", key="clear_button")
    
    # Initialize session state for canvas key to force refresh
    if 'canvas_key' not in st.session_state:
        st.session_state.canvas_key = 0
    
    # Increment canvas key when clear button is pressed to force refresh
    if clear_canvas:
        st.session_state.canvas_key += 1
        st.rerun()
    
    canvas_result = st_canvas(
        fill_color="black",  # Background
        stroke_width=20,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state.canvas_key}",
    )
    
    if canvas_result.image_data is not None:
        # Check if canvas has any drawing (not just black pixels)
        img_array = np.array(canvas_result.image_data)
        if np.any(img_array[:, :, :3] > 0):  # Check if there are any non-black pixels
            try:
                # Process the canvas image
                img = Image.fromarray(canvas_result.image_data)
                img = img.convert('L')  # Grayscale
                img = img.resize((28, 28))  # Resize to model input shape
                st.image(img, caption='Drawn Image (Resized)', use_container_width=True)
                
                # Convert to numpy array and preprocess
                image_array = np.array(img)
                processed_image = preprocess_data(image_array)
                processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension
                
                # Make prediction
                prediction = model.predict(processed_image)
                predicted_digit = np.argmax(prediction[0])
                confidence = np.max(prediction[0]) * 100
                
                # Display results
                st.subheader('Prediction')
                st.write(f"Predicted Digit: **{predicted_digit}**")
                st.write(f"Confidence: **{confidence:.2f}%**")
                
                # Optional: Show probability distribution
                if st.checkbox("Show probability distribution (Draw)"):
                    st.bar_chart(prediction[0])
            
            except Exception as e:
                st.error(f"Error processing drawn image: {str(e)}.")

# Additional info section
st.sidebar.title('Model Details')
st.sidebar.write("""
- **Architecture**: Flatten -> Dense(128, ReLU) -> Dense(32, ReLU) -> Dense(10, Softmax)
- **Training**: 15 epochs on MNIST, ~97.5% validation accuracy (from notebook)
- **Input**: 28x28 grayscale images, scaled to [0,1]
- **Loaded from**: mnist_pipeline.pkl (model extracted)
- **Note**: Install 'streamlit-drawable-canvas' via pip for the drawing feature.
""")