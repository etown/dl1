from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Router, Mount
from starlette.staticfiles import StaticFiles
from starlette.responses import FileResponse
from binascii import a2b_base64
from io import BytesIO
import os
import uvicorn
from fastai import *
from fastai.vision import *

classes = ['Anger', 'Disgust', 'Surprise', 'Sadness', 'Happiness', 'Neutral', 'Contempt', 'Fear']

data = ImageDataBunch.single_from_classes('', classes, tfms=get_transforms(), size=196).normalize(imagenet_stats)
learner = create_cnn(data, models.resnet34)
learner.load('gokul-sentiment-stage-5n')

app = Router(routes=[
    Mount('/static', app=StaticFiles(directory='static')),
])
@app.route('/')
async def homepage(request):
    return FileResponse('static/index.html')


@app.route('/face', methods=["GET","POST"])
async def face(request):
    body = await request.form()
    binary_data = a2b_base64(body['imgBase64'])
    img = open_image(BytesIO(binary_data))
    _,_,losses = learner.predict(img)
    analysis = {
        "predictions": dict(sorted(
            zip(learner.data.classes, map(float, losses)),
            key=lambda p: p[1],
            reverse=True
        ))}
    return JSONResponse(analysis)


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
