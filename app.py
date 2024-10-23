import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.utils import load_img, img_to_array
import streamlit as st

# Load the model
model = load_model('cat_dog_clf.keras')

def preprocess_image(image_path, target_size=(160, 160)):
    """
    Preprocesses the image for prediction.
    
    Args:
        image_path (str): Path to the image file.
        target_size (tuple): The target size for the image (width, height).
    
    Returns:
        np.ndarray: The preprocessed image ready for prediction.
    """
    # Load the image with the specified target size
    image = load_img(image_path, target_size=target_size)
    
    # Convert the image to a NumPy array
    image_array = img_to_array(image)
    
    # Scale the pixel values to the range [0, 1]
    image_array = image_array / 255.0
    
    # Expand dimensions to match the model's input shape (1, height, width, channels)
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def predict_image(image_path):
    """
    Predicts the class of an image using the trained model.
    
    Args:
        image_path (str): Path to the image file.
    
    Returns:
        str: Predicted class label.
    """
    # Preprocess the image
    processed_image = preprocess_image(image_path)
    
    # Make prediction
    predictions = model.predict(processed_image)
    
    # Convert predictions to binary labels (0 for cats, 1 for dogs)
    predicted_label = tf.where(predictions < 0.5, 0, 1).numpy()[0][0]
    
    # Return the class name based on predicted label
    return 'Dog' if predicted_label == 1 else 'Cat'

def main():
    st.title('Cat or Dog Classifier')
    st.write('Upload an image of a cat or dog and we will tell you which one it is')
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        image_path = f"temp_{uploaded_file.name}"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write("Classifying...")
        
        # Predict the image
        prediction = predict_image(image_path)
        st.write(f'The image is classified as: **{prediction}**')

if __name__ == '__main__':
    main()