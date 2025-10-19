"""
Part 3: Ethics & Optimization

This module addresses ethical considerations and provides optimization strategies
for the ML models developed in this project.
"""

def analyze_mnist_biases():
    """Analyze potential biases in the MNIST dataset and model."""
    print("\n=== MNIST Model Ethical Analysis ===")
    print("""
    Potential Biases:
    1. Dataset Bias:
       - Primarily contains Latin/Arabic numerals (0-9)
       - Handwriting styles may be biased towards certain demographics
       - Limited representation of different writing styles and variations
    
    2. Mitigation Strategies:
       a) Data Augmentation:
          - Random rotations, translations, and scaling
          - Adding noise to simulate different writing styles
          - Elastic deformations to mimic natural handwriting variations
          
       b) TensorFlow Fairness Indicators:
          - Install: pip install fairness-indicators
          - Evaluate model performance across different subgroups
          - Monitor metrics like false positive/negative rates per digit
          - Identify and address performance disparities
          
       c) Model Architecture:
          - Add dropout layers to prevent overfitting to specific styles
          - Use batch normalization for better generalization
          - Consider transfer learning from models trained on more diverse datasets
          
       d) Evaluation:
          - Test on diverse handwriting datasets
          - Monitor model performance on edge cases
          - Implement continuous monitoring in production
    """)

def analyze_reviews_biases():
    """Analyze potential biases in the Amazon Reviews model."""
    print("\n=== Amazon Reviews Model Ethical Analysis ===")
    print("""
    Potential Biases:
    1. Language and Cultural Bias:
       - Primarily English-language reviews
       - Cultural context may not be globally representative
       - Sentiment expressions vary across cultures
    
    2. Mitigation Strategies:
       a) spaCy's Rule-based Systems:
          - Custom tokenization rules for different languages
          - Rule-based pre-processing of text data
          - Entity recognition for identifying named entities
          
       b) Data Collection:
          - Include reviews from diverse demographics
          - Balance representation across different product categories
          - Consider regional language variations
          
       c) Sentiment Analysis:
          - Use domain-specific lexicons
          - Implement context-aware sentiment analysis
          - Consider cultural nuances in sentiment expression
          
       d) Fairness Evaluation:
          - Monitor performance across different demographic groups
          - Check for biased language patterns
          - Implement feedback loops for continuous improvement
    """)

def troubleshoot_tensorflow():
    """Common TensorFlow issues and solutions."""
    print("\n=== Common TensorFlow Issues and Solutions ===")
    print("""
    1. Dimension Mismatch Error:
       - Cause: Input shape doesn't match expected shape
       - Solution: Check input shapes using model.summary()
       - Example Fix: 
           # Before:
           # model.add(Dense(64, input_shape=(784,)))
           # After:
           # model.add(Dense(64, input_shape=(28, 28, 1)))
           # model.add(Flatten())
    
    2. Incorrect Loss Function:
       - Cause: Using wrong loss function for the task
       - Solution: 
           - For binary classification: 'binary_crossentropy'
           - For multi-class: 'categorical_crossentropy'
           - For regression: 'mse' or 'mae'
    
    3. Vanishing/Exploding Gradients:
       - Symptoms: Model doesn't learn or produces NaNs
       - Solutions:
           - Use batch normalization
           - Try different weight initializers
           - Use gradient clipping
    """)

def deployment_guide():
    """Guide for deploying the MNIST classifier using Streamlit/Flask."""
    print("\n=== Model Deployment Guide ===")
    print("""
    Streamlit Deployment:
    
    1. Install Streamlit:
       pip install streamlit
    
    2. Create app.py:
       ```python
       import streamlit as st
       import numpy as np
       from PIL import Image
       import tensorflow as tf
       
       # Load the trained model
       model = tf.keras.models.load_model('mnist_cnn_model.h5')
       
       st.title('MNIST Digit Classifier')
       st.write('Upload an image of a digit (0-9)')
       
       uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
       
       if uploaded_file is not None:
           # Preprocess the image
           image = Image.open(uploaded_file).convert('L')
           image = image.resize((28, 28))
           img_array = np.array(image) / 255.0
           img_array = img_array.reshape(1, 28, 28, 1)
           
           # Make prediction
           prediction = model.predict(img_array)
           predicted_digit = np.argmax(prediction)
           
           st.image(image, caption=f'Predicted: {predicted_digit}', use_column_width=True)
           st.write(f'Confidence: {np.max(prediction) * 100:.2f}%')
       ```
    
    3. Run the app:
       streamlit run app.py
    
    4. Deploy to Streamlit Cloud:
       - Push code to GitHub
       - Go to share.streamlit.io
       - Connect your repository
       - Deploy!
    """)

if __name__ == "__main__":
    print("AI Models: Ethical Considerations and Optimization")
    print("-" * 50)
    analyze_mnist_biases()
    analyze_reviews_biases()
    troubleshoot_tensorflow()
    deployment_guide()
