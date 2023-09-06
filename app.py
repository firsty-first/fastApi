from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import json
from ultralytics import YOLO
import requests
from PIL import Image
from io import BytesIO

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model = YOLO('Finalmodel.onnx', task='detect')

def count_plastic_from_image(image_path):
    img =Image.open(BytesIO(requests.get(image_path).content))
    img = img.resize([2024, 1024])
    load = model.predict(img, save=True, imgsz=1024, conf=0.204)
    count = len(load[0])
    return count


class model_input(BaseModel):
    imgUrl : str



@app.get('/')
def main():
    return {'message': 'Welcome to Render Fast Api'}

@app.post('/prediction')
def getPrediction(input_parameters : model_input):
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)

    imgUrl = str(input_dictionary['imgUrl'])

    prediction = count_plastic_from_image(imgUrl)
    # return "Total number of plastic in the image is: ".format(prediction)
    return f"Total number of plastic in the image is: {prediction}"

