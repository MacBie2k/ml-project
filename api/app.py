from io import BytesIO

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, abort
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


def prepare(img):
    img_size = 224
    scaled_array = cv2.resize(img, (img_size, img_size))
    return scaled_array.reshape(-1, img_size, img_size, 3)


def loadModel(modelpath):
    return tf.keras.models.load_model(modelpath)


def predict(model, img):
    return model.predict([prepare(img)])


model = loadModel(
    ".../ml-project/model/garbarge-types-1687543439.model")


@app.route('/garbage', methods=['POST'])
def sort_garbage():
    if 'file' not in request.files:
        abort(400, 'No file part in the request.')

    file = request.files['file']
    if file.filename == '':
        abort(400, 'No file selected for uploading')

    in_memory_file = BytesIO()
    file.save(in_memory_file)
    data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        abort(400, 'Could not open or find the image.')

    categories = ["cardboard", "glass", "metal", "paper", "plastic"]

    prediction = predict(model, img)
    index = np.argmax(prediction)
    print(categories[np.argmax(prediction)])
    response = {
        'prediction': prediction.tolist(),
        'predicted_index': int(index),
        'predicted_label': categories[index]
    }
    return response, 200

if __name__ == '__main__':
    app.run()
