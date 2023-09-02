from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import os
app = FastAPI()

# Import your YOLO model and define it here
# Example: from yolo_module import YOLO
model = YOLO('Finalmodel.onnx', task='detect')

def count_plastic_from_image(image_path):
    # Perform object detection on the image
    load = model.predict(image_path, save=True, imgsz=1024, conf=0.204)
    count = len(load[0])
    return count


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
