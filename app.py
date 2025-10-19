import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# Set page config
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="wide"
)

def load_model():
    """Load the trained MNIST model."""
    try:
        # Try multiple possible model locations
        possible_paths = [
            os.path.join('models', 'mnist_cnn_model.h5'),  # Local models directory
            os.path.join('..', 'models', 'mnist_cnn_model.h5'),  # Parent directory's models folder
            'mnist_cnn_model.h5'  # Current directory
        ]
        
        model = None
        for model_path in possible_paths:
            try:
                if os.path.exists(model_path):
                    st.info(f"Loading model from: {os.path.abspath(model_path)}")
                    model = tf.keras.models.load_model(model_path)
                    return model
            except Exception as e:
                st.warning(f"Failed to load model from {model_path}: {e}")
        
        # If we get here, no model was loaded successfully
        st.error(f"Could not find model in any of these locations: {', '.join(possible_paths)}")
        return None
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image):
    """Preprocess the uploaded image for prediction."""
    try:
        # Convert to grayscale and resize
        img = ImageOps.grayscale(image)
        img = img.resize((28, 28))
        
        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        
        # Reshape for model input
        img_array = img_array.reshape(1, 28, 28, 1)
        return img_array, img
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, None

def main():
    st.title("🔢 MNIST Digit Classifier")
    st.write("Upload an image of a handwritten digit (0-9) and the model will predict the digit.")
    
    # Load model
    model = load_model()
    if model is None:
        st.error("Failed to load the model. Please ensure the model file exists.")
        return
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        # Sample images
        st.markdown("### Or try a sample image:")
        sample_cols = st.columns(3)
        sample_images = [f"sample_{i}.png" for i in range(10)]
        
        for i, sample in enumerate(sample_images):
            if i % 3 == 0:
                sample_cols = st.columns(3)
            
            with sample_cols[i % 3]:
                if st.button(f"Sample {i}"):
                    try:
                        # Try to load sample image
                        img_path = os.path.join("data", "samples", sample)
                        if os.path.exists(img_path):
                            image = Image.open(img_path)
                            img_array, processed_img = preprocess_image(image)
                            
                            if img_array is not None:
                                # Make prediction
                                prediction = model.predict(img_array)
                                predicted_digit = np.argmax(prediction[0])
                                confidence = np.max(prediction[0]) * 100
                                
                                # Display results
                                with col2:
                                    st.header("Prediction Result")
                                    st.image(processed_img, caption=f"Sample {i}", use_column_width=True)
                                    st.success(f"Predicted Digit: {predicted_digit}")
                                    st.info(f"Confidence: {confidence:.2f}%")
                                    
                                    # Show prediction probabilities
                                    fig, ax = plt.subplots()
                                    ax.bar(range(10), prediction[0] * 100)
                                    ax.set_xticks(range(10))
                                    ax.set_xlabel('Digit')
                                    ax.set_ylabel('Confidence (%)')
                                    ax.set_title('Prediction Probabilities')
                                    st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error processing sample image: {e}")
    
    # Process uploaded file
    if uploaded_file is not None:
        try:
            # Open and process the image
            image = Image.open(uploaded_file)
            img_array, processed_img = preprocess_image(image)
            
            if img_array is not None:
                # Make prediction
                prediction = model.predict(img_array)
                predicted_digit = np.argmax(prediction[0])
                confidence = np.max(prediction[0]) * 100
                
                # Display results in the right column
                with col2:
                    st.header("Prediction Result")
                    st.image(processed_img, caption="Uploaded Image", use_column_width=True)
                    st.success(f"Predicted Digit: {predicted_digit}")
                    st.info(f"Confidence: {confidence:.2f}%")
                    
                    # Show prediction probabilities
                    fig, ax = plt.subplots()
                    ax.bar(range(10), prediction[0] * 100)
                    ax.set_xticks(range(10))
                    ax.set_xlabel('Digit')
                    ax.set_ylabel('Confidence (%)')
                    ax.set_title('Prediction Probabilities')
                    st.pyplot(fig)
                    
        except Exception as e:
            st.error(f"Error processing image: {e}")

if __name__ == "__main__":
    main()
