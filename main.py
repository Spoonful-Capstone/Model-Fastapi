import io
import numpy as np
from typing import Union
import pandas as pd
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model("model.h5")

class_id = ['Ayam Betutu', 'Beberuk Terong', 'Coto Makassar', 'Gudeg', 'Kerak Telor', 'Mie Aceh', 'Nasi Kuning', 
            'Nasi Pecel', 'Papeda', 'Pempek', 'Peuyeum', 'Rawon', 'Rendang', 'Sate Madura', 'Serabi', 'Soto Banjar',
            'Soto Lamongan', 'Tahu Sumedang']
nutri_df = pd.DataFrame({
    'Kalori': [425, 65, 289, 127, 452, 116, 150, 230, 58, 39, 175, 119, 195, 34, 108, 100, 312, 35],
    'Protein': [22.60, 0.79, 25.40, 1.82, 20.11, 3.34, 0.27, 6.58, 0.04, 2.52, 2.27, 9.60, 19.68, 2.93, 1.59, 4.66, 24.01, 2.12],
    'Lemak': [34.80, 3.85, 15.25, 4.71, 15.81, 3.58, 0.27, 9.33, 0.01, 1.04, 0.50, 7.40, 11.07, 2.22, 1.80, 6.71, 14.92, 2.42],
    'Karbohidrat': [6.32, 7.09, 12.92, 22.54, 55.58, 17.96, 2.99, 31.74, 14.03, 4.72, 39.90, 3.48, 4.49, 0.73, 21.05, 5.29, 19.55, 1.30]
})
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
    
    cal = json.dumps(float(nutri_df['Kalori'].iloc[result.index(m)]))
    prot = json.dumps(float(nutri_df['Protein'].iloc[result.index(m)]))
    fat = json.dumps(float(nutri_df['Lemak'].iloc[result.index(m)]))
    carb = json.dumps(float(nutri_df['Karbohidrat'].iloc[result.index(m)]))
        
    return JSONResponse(content={
        "Name": class_id[result.index(m)], 
        "Nutritions": {
            "Calories": cal, 
            "Proteins": prot,
            "Fat": fat,
            "Carbohydrate": carb}
        }, status_code=200)
    
