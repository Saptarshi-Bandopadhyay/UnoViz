from flask import Flask, redirect, url_for, render_template, request, Response
import cv2 as cv
import numpy as np
import os
import src.pipeline.predict_pipeline as model
from werkzeug.utils import secure_filename

net2 = cv.dnn.readNetFromCaffe(
    "./artifacts/deploy.prototxt.txt", "./artifacts/res10_300x300_ssd_iter_140000.caffemodel")

UPLOAD_FOLDER = './artifacts/faces'
ALLOWED_EXTENSIONS = {'jpg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

cap = cv.VideoCapture(0)

Face_Folder = './artifacts/faces'
app.config['FACE_FOLDER'] = Face_Folder


def generate_frames():
    while True:
        success, image = cap.read()
        if success:
            (h, w) = image.shape[:2]
            blob = cv.dnn.blobFromImage(cv.resize(image, (300, 300)), 1.0,
                                        (300, 300), (104.0, 177.0, 123.0))
            model.net.setInput(blob)
            detections = model.net.forward()
            i = np.argmax(detections[0, 0, :, 2])
            confidence = np.max(detections[0, 0, :, 2])
            if confidence > 0.6:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face = image[startY:endY, startX:endX, :]
                try:
                    prediction = model.who(face)
                    text = prediction
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv.rectangle(image, (startX, startY), (endX, endY),
                                 (0, 0, 255), 2)
                    cv.putText(image, text, (startX, y),
                               cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                except:
                    pass
            ret, buffer = cv.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

        else:
            break


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET'])
@app.route('/index', methods=['GET'])
def index():
    face_names = model.names[:-1]
    return render_template('index.html', names=face_names)


@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            # flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            # flash('No selected file')
            return redirect(url_for('index'))
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            image = cv.imread(file_path)
            (h, w) = image.shape[:2]
            blob = cv.dnn.blobFromImage(cv.resize(image, (300, 300)), 1.0,
                                        (300, 300), (104.0, 177.0, 123.0))
            net2.setInput(blob)
            detections = net2.forward()
            i = np.argmax(detections[0, 0, :, 2])
            confidence = np.max(detections[0, 0, :, 2])
            if confidence > 0.7:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face = image[startY:endY, startX:endX, :]
                try:
                    a = cv.imwrite(file_path, face)
                    model.people = model.get_face_embeddings()
                    model.names = [name.split('.', 1)[0] for name in os.listdir(
                        "./artifacts/faces/")] + ["Unknown"]
                except:
                    print("face could not be detected")
            return redirect(url_for('index'))


@app.route('/video')
def video():
    frame = generate_frames()
    return Response(frame, mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/artifacts/faces/<name>')
def image(name):
    image = open(os.path.join(app.config['FACE_FOLDER'], name), "rb").read()
    return Response(image, content_type="image/jpg")


if __name__ == '__main__':
    app.run(debug=True)
