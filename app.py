from flask import Flask, redirect, url_for, render_template, request, Response
import cv2 as cv
import numpy as np
import src.pipeline.predict_pipeline as model

app = Flask(__name__)

cap = cv.VideoCapture(0)


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
                               cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                except:
                    pass
            ret, buffer = cv.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

        else:
            break


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    frame = generate_frames()
    return Response(frame, mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
