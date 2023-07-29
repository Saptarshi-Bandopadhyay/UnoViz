from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
import cv2 as cv
import os

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


def get_face_embeddings():
    people = []
    faces_path = "./artifacts/faces/"
    for person in os.listdir(faces_path):
        people.append(embedding.predict(
            preprocess_file(os.path.join(faces_path, person))))
    return people


people = get_face_embeddings()


def preprocess_image(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (224, 224))
    image = image[None, :, :, :]
    return image


def who(face):
    face = preprocess_image(face)
    pred = embedding.predict(face, verbose=0)
    distance = []
    for individual in people:
        distance.append(np.sum(np.square(pred-individual), axis=-1))
    name = ["Saptarshi", "Shubhradeep", "Trijeta", "Unknown"]
    if np.min(distance) > 1.8:
        return name[3]
    return name[np.argmin(distance)]
