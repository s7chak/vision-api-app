from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse
from starlette.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
from fastai.vision import *
import torch
from pathlib import Path
from io import BytesIO
import sys
import uvicorn
import aiohttp
import asyncio


async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()


app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='static'))

path = Path(__file__).parent
threat_images_path = Path("/tmp")
classes = ['gun','knife','bomb']
threat_fnames = [
    "images/{}_1.jpg".format(c)
    for c in classes
]

# model_file_name='threat_model'






threat_data = ImageDataBunch.single_from_classes(path, classes, ds_tfms=get_transforms(), size=150).normalize(imagenet_stats)
# from_name_re(
#     threat_images_path,
#     threat_fnames,
#     r"/([^/]+)_\d+.jpg$",
#     ds_tfms=get_transforms(),
#     size=224,
# )

# threat_learner = cnn_learner(threat_data, models.resnet34)
# # threat_learner.load(model_file_name)
# threat_learner.model.load_state_dict(
#     torch.load("threat_model.pth", map_location="cpu")['model']
# )


# threat_learner = cnn_learner(threat_data, models.resnet34)
# threat_learner.model.load_state_dict(
#     torch.load("models/threat_model.pth", map_location="cpu")['model']
# )


threat_learner=load_learner('.')

@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    bytes = await (data["file"].read())
    return predict_image_from_bytes(bytes)


@app.route("/classify-url", methods=["GET"])
async def classify_url(request):
    bytes = await get_bytes(request.query_params["url"])
    return predict_image_from_bytes(bytes)


def predict_image_from_bytes(bytes):
    img = open_image(BytesIO(bytes))
    _, class_, losses = threat_learner.predict(img)
    pred=sorted(
            zip(threat_learner.data.classes, map(float, losses)),
            key=lambda p: p[1], reverse=True
        )

    print(pred)

    result_html1 = path/'static'/'result1.html'
    result_html2 = path/'static'/'result2.html'
    
    result_html = str(result_html1.open().read() +"It's a "+pred[0][0] + result_html2.open().read())
    return HTMLResponse(result_html)

    # return HTMLResponse("""
    #     <body>
    #     <h3 align="center">The image shows a """+str(pred[0][0])+"""<h3>
    #     </body>
    # """)



@app.route("/")
def form(request):
    index_html = path/'static'/'index.html'
    return HTMLResponse(index_html.open().read())
    # return HTMLResponse("""
    #     <h3>This app will classify a hazardous object - gun, knife or a bomb<h3>
    #     <form action="/upload" method="post" enctype="multipart/form-data">
    #         Select image to upload:
    #         <input type="file" name="file">
    #         <input type="submit" value="Upload Image">
    #     </form>
    #     Or submit a URL:
    #     <form action="/classify-url" method="get">
    #         <input type="url" name="url">
    #         <input type="submit" value="Fetch and analyze image">
    #     </form>
    # """)


@app.route("/form")
def redirect_to_homepage(request):
    return RedirectResponse("/")


if __name__ == "__main__":
    if "serve" in sys.argv:
        uvicorn.run(app, host="0.0.0.0", port=8080, debug=True)
