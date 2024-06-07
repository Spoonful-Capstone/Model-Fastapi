import io
import numpy as np
from typing import Union
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model("model.h5")
pil_image = Image.open("Screenshot 2024-06-07 153654.jpg")
pil_image =pil_image.resize((256,256))
img_array = np.array(pil_image)
img_array = img_array/255.0    
img_array = np.expand_dims(img_array, axis=0)
prediction = model.predict(img_array)

class_id = ['Ayam Betutu', 'Beberuk Terong', 'Coto Makassar', 'Gudeg', 'Kerak Telor', 'Mie Aceh', 'Nasi Kuning', 
            'Nasi Pecel', 'Papeda', 'Pempek', 'Peuyeum', 'Rawon', 'Rendang', 'Sate Madura', 'Serabi', 'Soto Banjar',
            'Soto Lamongan', 'Tahu Sumedang']
result = []

@app.post("/predict-image/")
async def predict_image(file: UploadFile = File(...)):    
    contents = await file.read()
    
    pil_image = Image.open(io.BytesIO(contents))
    pil_image =pil_image.resize((256,256))
    img_array = np.array(pil_image)
    img_array = img_array/255.0    
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    
    for i in range(len(prediction[0])):
        result.append(float(prediction[0][i]))

    m = max(result)
        
    return {"Prediction": class_id[result.index(m)]}

