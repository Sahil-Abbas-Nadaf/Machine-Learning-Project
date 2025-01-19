try:
    import streamlit as st
    import pandas as pd
    import matplotlib.pyplot as plt
    import os 
    import sys
    from io import BytesIO, StringIO
    import numpy as np
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    import warnings
    warnings.filterwarnings('ignore')
    print("All Modules Loaded...")
except Exception as e:
    print("Some Modules are Missing : {}".format(e))

def process(file, show_file):
    content = file.getvalue()

    if isinstance(file, BytesIO):
        show_file.image(file)


    # Load the trained model
    model = load_model(r'C:\Users\nadaf\OneDrive\Documents\Potato Leaf Disease Prediction Project\potato_model.h5')  # Updated model path

    # Preprocess the image
    IMG_SIZE = (128, 128)  # Must match the size used during training

    def preprocess_image(file):
        img = load_img(file, target_size=IMG_SIZE)  # Load and resize image
        img_array = img_to_array(img)  # Convert image to array
        img_array = img_array / 255.0  # Rescale to 0-1 range
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array

    image = preprocess_image(file)

    # Make prediction
    prediction = model.predict(image)

    # Class labels (ensure they match the model's training)
    class_labels = ['Early_Blight', 'Healthy', 'Late_Blight']

    # Get the predicted class index
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_class_label = class_labels[predicted_class_index]

    # Output the prediction
    st.write(f"The image is predicted to be: {predicted_class_label}")
        
    file.close()

def main():
    st.title("Potato Leaf Disease Detection and Prediction")
    file = st.file_uploader("Choose a JPG file", type = "JPG")
    show_file = st.empty()

    if not file: 
        show_file.info("Please upload a file : {}".format("JPG"))
        return

    bt = st.button("Analyze")

    if bt == True:
        process(file, show_file)


if __name__ == "__main__":
    main()