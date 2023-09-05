from fastapi import FastAPI, UploadFile, File,HTTPException
from urllib import request
from io import BytesIO
from PIL import Image
from ultralytics import YOLO
import os
app = FastAPI()

# Import your YOLO model and define it here
# Example: from yolo_module import YOLO
model = YOLO('Finalmodel.onnx', task='detect')

@app.get("/hello")
async def hello():
    return {"message": "Hello, FastAPI!"}

def count_plastic_from_image(image_path):
    # Perform object detection on the image
    load = model.predict(image_path, save=True, imgsz=1024, conf=0.204)
    count = len(load[0])
    return count
def count_plastic_from_image_uri(image_uri):
    try:
        # Download the image from the provided URI
        response = request.urlopen(image_uri)
        image_data = response.read()

        # Open the downloaded image using PIL (Python Imaging Library)
        image = Image.open(BytesIO(image_data))
        
        # Convert the image to a format that the YOLO model can process
        img_array = np.array(image)
        
        # Perform object detection on the image
        load = model.predict(img_array, save=True, imgsz=1024, conf=0.204)
        count = len(load[0])
        return count
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
@app.post("/count_plastic_uri/")
async def count_plastic_uri(image_uri: str):
    try:
        # Perform object detection using your local method
        count = count_plastic_from_image_uri(image_uri)

        return {"count": count}
    except Exception as e:
        return {"error": str(e)}


# @app.post("/")
# def home():
#     return "welcome to fast api"

@app.post("/count_plastic/")
async def count_plastic(file: UploadFile):
    try:
        # Ensure the uploaded file is an image
        if file.content_type.startswith('image'):
            # Save the uploaded image to a temporary file
            with open("temp_image.jpg", "wb") as temp_image:
                temp_image.write(file.file.read())

            # Perform object detection using your local method
            count = count_plastic_from_image("temp_image.jpg")

            # Delete the temporary image file
            os.remove("temp_image.jpg")

            return {"count": count}
        else:
            return {"error": "Invalid file format"}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
# uvicorn app.py:app --host 0.0.0.0 --port 8000
