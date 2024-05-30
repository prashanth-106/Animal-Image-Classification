# Packages required for Image Classification
import base64
import streamlit as st
from PIL import Image
import requests
# Necessary imports from keras
# VGG16, A pretrained model in keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16

# Function definitions
# To set background
def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.
    Parameters:
        image_file (str): The path to the image file to be used as the background.
    Returns:
        None
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)
# To predict the image
def predict(image1): 
    model = VGG16()
    image = load_img(image1, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # predict the probability across all output classes
    yhat = model.predict(image)
    # convert the probabilities to class labels
    label = decode_predictions(yhat)
    # retrieve the most likely result, e.g. highest probability
    label = label[0][0]
    return label 
# To get image with url
def get_image(url):
    img = requests.get(url)
    file = open("sample_image.jpg", "wb")
    file.write(img.content)
    file.close()
    img_file_name = 'sample_image.jpg'
    return img_file_name

# Main driver
# set background
set_background('./bgs/bg.png')
st.title("Animal Image Classification")
url = st.text_input("Enter Image Url:")
if url:
    image = get_image(url)
    st.image(image)
    classify = st.button("classify image")
    if classify:
        st.write("Classifying...")
        label = predict(image)
        st.write('%s (Accuracy : %.2f%%)' % (label[1], label[2]*100))