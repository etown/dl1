# Real-time emotion evaluator

### To run locally:
  - Install fastai, starlette and uvicorn
  - `python server.py`
  - Go to http://localhost:8000/ (many browsers require https for webRTC)

## Inference Server

Once we had a trained model, it was time to share it with the world. Creating a server was pretty simple. We used the [Starlette](https://www.starlette.io/) framework. Starlette was so easy to use and has such great documentation that I hardly missed node.js at all.

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


