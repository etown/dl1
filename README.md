# Real-time emotion evaluator
[![video](https://img.youtube.com/vi/XhkCfwK7OBA/0.jpg)](https://www.youtube.com/watch?v=XhkCfwK7OBA)


### Live demo:
[fastai.mollyai.com](https://fastai.mollyai.com/)

### To run locally:
  - Install fastai, starlette and uvicorn
  - `python server.py`
  - Go to http://localhost:8000/ (many browsers require https for webRTC)

## The Model
In order to distinguish different facial expressions, we chose to train the model on the [AffectNet](http://mohammadmahoor.com/wp-content/uploads/2017/08/AffectNet_oneColumn-2.pdf) dataset. This is a phenomenal data set of 450,000 manually annotated face images.

For example:

![Image of faces](https://github.com/etown/dl1/raw/master/face/example_faces.png)

For much more information on training, please check out the [notebook](https://github.com/etown/dl1/blob/master/face/Facial_Emotion_Recogonition.ipynb).

## Inference Server

Once we had a trained model, it was time to share it with the world. Creating a server was pretty simple. We used the [Starlette](https://www.starlette.io/) framework.

First, in our server file outside of any route, we had to create a [Learner](https://docs.fast.ai/basic_train.html#Learner) similar to the one we used to train our model. So, we (1) define our classes in the same way we did when training, (2) instantiate an ImageDataBunch and (3) instantiate a Learner from that ImageDataBunch.

```python
classes = ['Anger', 'Disgust', 'Surprise', 'Sadness', 'Happiness', 'Neutral', 'Contempt', 'Fear']

data = ImageDataBunch.single_from_classes('', classes, tfms=get_transforms(), size=196).normalize(imagenet_stats)
learner = create_cnn(data, models.resnet34)
learner.load('sentiment-model')
```

Note that when calling `ImageDatabunch.single_from_classes`, the first argument is the path to your model. The function expects the model to be in a `models` directory under that path. So, if your `models` directory is in the same directory as your server file, you can just use an empty string for your path as we did here. Then when calling `learner.load('sentiment-model')`, you will be loading `./models/sentiment-model.pth`.

Now that we have our weights loaded and ready to make inferences, we set up a route to accept images from a client and return a set of predictions.

```python
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
```
Every time the `/face` route is hit, our Learner will evaluate the image sent in the request. From calling `learner.predict(img)` we get a tensor of probabilities for each class in the same order of the classes as defined above. We then zip these predictions to the class names and order by the probabilities in descending order and transform into a dictionary (for ease of use by a javascript client). So, if we had a happy image, the resulting analysis might look like:

`{'predictions': {'Happiness': 29.584692001342773, 'Sadness': 1.511526346206665, 'Neutral': 0.5242078900337219, 'Fear': 0.33813756704330444, 'Contempt': 0.29101505875587463, 'Surprise': 0.282543420791626, 'Anger': 0.1381775140762329, 'Disgust': 0.10192008316516876}}`

## Web Client
We want the recognition to be as smooth and real-time as possible. We also want to distribute on the web. We can take advantage of webRTC to capture frames from the camera video stream, send them to our inference server and then display the results. However, in order to maximize the responsiveness, we need to send images fairly frequently. In order to reduce the network overhead, we use [trackingjs](https://trackingjs.com/) to first track the face, crop it, and then finally only send the face to the server. By first cropping the face, we can drastically reduce the latency.

We get the cropped face, draw it to a different canvas, and then use `toDataURL()` to send the base64 face image to the server.

