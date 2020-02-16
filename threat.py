from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse
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

path = Path(__file__).parent
threat_images_path = Path("/tmp")
threat_fnames = [
    "images/{}_1.jpg".format(c)
    for c in [
        "gun",
        "knife",
        "bomb"
    ]
]
classes = ['gun','knife','bomb']
model_file_name='threat_model'

threat_data = ImageDataBunch.single_from_classes(path, classes, ds_tfms=get_transforms(), size=150).normalize(imagenet_stats)
# from_name_re(
#     threat_images_path,
#     threat_fnames,
#     r"/([^/]+)_\d+.jpg$",
#     ds_tfms=get_transforms(),
#     size=224,
# )
threat_learner = cnn_learner(threat_data, models.resnet34)
threat_learner.load(model_file_name)
# threat_learner.model.load_state_dict(
#     torch.load("threat_model.pth", map_location="cpu")['model']
# )


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
    _,class_,losses = threat_learner.predict(img)

    print({
        "prediction": classes[class_.item()],
        "scores": sorted(
            zip(threat_learner.data.classes, map(float, losses)),
            key=lambda p: p[1],
            reverse=True
        )
    })

    prediction=sorted(
            zip(threat_learner.data.classes, map(float, losses)),
            key=lambda p: p[1])
    
    print(prediction)
    return HTMLResponse(
        """ <h1>It's a """+prediction[0][0]+"""</h1>""")


@app.route("/")
def form(request):
    return HTMLResponse(
        """
        <h2>Threat Object Detector<h2>
        <form action="/upload" method="post" enctype="multipart/form-data">
            Select image to upload:
            <input type="file" name="file">
            <input type="submit" value="Upload Image">
        </form>
        Or submit a URL:
        <form action="/classify-url" method="get">
            <input type="url" name="url">
            <input type="submit" value="Fetch and analyze image">
        </form>
    """)


@app.route("/form")
def redirect_to_homepage(request):
    return RedirectResponse("/")


if __name__ == "__main__":
    if "serve" in sys.argv:
        uvicorn.run(app, host="0.0.0.0", port=8008)
