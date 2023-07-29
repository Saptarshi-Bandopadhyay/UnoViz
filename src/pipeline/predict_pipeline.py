from keras.models import load_model
import numpy as np
from keras.utils import image_dataset_from_directory
import tensorflow as tf
import cv2 as cv
import os
import shutil

Model = load_model(".\\artifacts\siamese_network", compile=False)
embedding = Model.layers[3]
net = cv.dnn.readNetFromCaffe(
    ".\\artifacts\deploy.prototxt.txt", ".\\artifacts\\res10_300x300_ssd_iter_140000.caffemodel")


def preprocess_file(file_path):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (224, 224))
    image = image[None, :, :, :]
    return image


Saptarshi = embedding.predict(
    preprocess_file(".artifacts/faces/Saptarshi.jpg"))
Shubhradeep = embedding.predict(
    preprocess_file(".artifacts/faces/Shubhradeep.jpg"))
Trijeta = embedding.predict(preprocess_file(".artifacts/faces/Trijeta.jpg"))


def preprocess_image(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (224, 224))
    image = image[None, :, :, :]
    return image


def who(face):
    face = preprocess_image(face)
    pred = embedding.predict(face, verbose=0)
    distance = []
    distance.append(np.sum(np.square(pred-Saptarshi), axis=-1))
    distance.append(np.sum(np.square(pred-Shubhradeep), axis=-1))
    distance.append(np.sum(np.square(pred-Trijeta), axis=-1))
    name = ["Saptarshi", "Shubhradeep", "Trijeta", "Unknown"]
    if np.min(distance) > 1.8:
        return name[3]
    return name[np.argmin(distance)]
