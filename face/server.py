from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Router, Mount
from starlette.staticfiles import StaticFiles
from binascii import a2b_base64
import requests
import os
import uvicorn

app = Router(routes=[
    Mount('/static', app=StaticFiles(directory='static')),
])
@app.route('/')
async def homepage(request):
    return JSONResponse({'hello': 'world'})


@app.route('/face', methods=["GET","POST"])
async def face(request):
    body = await request.form()
    binary_data = a2b_base64(body['imgBase64'])
    fd = open('image.png', 'wb')
    fd.write(binary_data)
    fd.close()
    params = {
        'returnFaceId': 'true',
        'returnFaceLandmarks': 'false',
        'returnFaceAttributes': 'emotion'
    }
    headers  = {'Ocp-Apim-Subscription-Key': os.environ['MSKEY'], "Content-Type": "application/octet-stream" }
    response = requests.post("https://westcentralus.api.cognitive.microsoft.com/face/v1.0/detect", params=params, headers=headers, data=binary_data)
    response.raise_for_status()
    analysis = response.json()
    
    return JSONResponse(analysis)



if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)