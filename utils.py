import tensorflow as tf
import streamlit as st
from fastapi import FastAPI
import uvicorn
from PIL import Image
import numpy as np
import keras
import keras_hub
model=keras.models.load_model(r"D:\NTI Tasks\cv project\resnet_model.keras")
def preprocessing_image(image):
    image=image.resize((32,32))
    image=np.array(image) /255.0
    image=np.expand_dims(image,axis=0)
    return image
def predict(input_image):
    input_image=preprocessing_image(input_image)
    predictions = model.predict(input_image)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100

    return predicted_class, confidence
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]