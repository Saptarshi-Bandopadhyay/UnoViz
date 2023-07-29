import cv2 as cv
import os
import random
import numpy as np

net = cv.dnn.readNetFromCaffe(
    "./artifacts/deploy.prototxt.txt", "./artifacts/res10_300x300_ssd_iter_140000.caffemodel")

input_path = "./artifacts/lfw_funneled"
train_output_path = "./artifacts/train"
test_output_path = "./artifacts/test"


def name_list(input_path):
    names = []

    for subdir in os.listdir(input_path):
        if len(os.listdir(os.path.join(input_path, subdir))) > 1:
            names.append(subdir)
    return names


def cropped_face(file_path):
    image = cv.imread(file_path)
    (h, w) = image.shape[:2]
    blob = cv.dnn.blobFromImage(cv.resize(image, (300, 300)), 1.0,
                                (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    i = np.argmax(detections[0, 0, :, 2])
    confidence = np.max(detections[0, 0, :, 2])
    if confidence > 0.6:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        face = image[startY:endY, startX:endX, :]

    return face


def make(output_path, folder):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for subdir in folder:
        if not os.path.exists(os.path.join(output_path, subdir)):
            os.makedirs(os.path.join(output_path, subdir))
        for file in os.listdir(os.path.join(input_path, subdir)):
            try:
                image = cropped_face(os.path.join(input_path, subdir, file))
                cv.imwrite(os.path.join(output_path, subdir, file), image)
            except:
                continue


def create():
    folders = name_list(input_path)
    random.seed(42)
    random.shuffle(folders)

    train_set = folders[100:]
    test_set = folders[:100]
    make(train_output_path, train_set)
    make(test_output_path, test_set)


if __name__ == '__main__':
    create()
