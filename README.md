# Threat Levels

My first attempt at a machine learning API, using a pre-calculated model trained using collected images from Google Search.


The model is `threat_model.pth` - a 91MB file.

The notebook `inaturalist-cats.ipynb` shows how I trained the model, using [fastai](https://github.com/fastai/fastai).

`threat.py` is a very tiny [Starlette](https://www.starlette.io/) API server which simply accepts file image uploads and runs them against the pre-calculated model.

It also accepts a URL to an image

The `Dockerfile` means the entire thing can be deployed to [Google App Engine]