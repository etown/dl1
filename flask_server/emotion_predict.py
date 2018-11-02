import cv2
from flask import Flask, abort, Response, render_template, request
import numpy as np
import datetime
import scipy.misc
from fastai import *
from fastai.vision import *
from PIL import Image as ImagePIL


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
app = Flask(__name__)


def load_model():
    path = Path("/tmp")
    classes = ['sad', 'happy', 'neutral',
               'angry', 'surprise', 'disgust', 'fear']
    empty_ds = ImageClassificationDataset([classes[0]], [classes[0]],
                                          classes)
    data = ImageDataBunch.create(empty_ds, empty_ds, path=path,
                                 ds_tfms=get_transforms(),
                                 size=196)
    learn = create_cnn(data, models.resnet34)
    learn.load('/tmp/models/stage-4')

    return learn


def get_faces(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im_copy = np.copy(im)
    gray_image = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, 1.25, 6)
    face_crop = []
    for f in faces:
        x, y, w, h = [v for v in f]
        face_crop.append(im_copy[y:y + h, x:x + w])
    return face_crop


def get_prediction(file_path):
    image = cv2.imread(file_path)
    faces = get_faces(image)
    img = ImagePIL.fromarray(faces[0])
    img = Image(pil2tensor(img).float().div_(255))
    losses = img.predict(model)
    return model.data.classes[np.argmax(losses)]


@app.route('/')
def serve_results():
    return render_template("FacialEmotion.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}

    if request.method == "POST":
        try:
            file_path = '/tmp/p/' + str(datetime.datetime.now()) + '.jpg'
            request.files['myFile'].save(file_path)
            data['emotion'] = get_prediction(file_path)
            print(data['emotion'])
        except:
            data['emotion'] = 'Try another pic'

    return Response(data['emotion'])


if __name__ == "__main__":
    model = load_model()
    app.run('0.0.0.0')
