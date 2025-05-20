from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import cv2

# Load your pre-trained model
model = load_model('my_model.h5')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Class index to label mapping
class_labels = {
    0: 'Castor',
    1: 'Catharanthus',
    2: 'Curry Leaf',
    3: 'Doddapatre',
    4: 'Mint',
    5: 'Neem',
    6: 'Palak(Spinach)',
    7: 'Papaya',
    8: 'Tamarind',
    9: 'Tulasi'
}

# Plant descriptions dictionary
plant_descriptions =  {
    "Castor": "(Ricinus communis) - Used as a laxative and to treat skin conditions.",
    "Catharanthus": "(Catharanthus roseus) - Traditionally used to treat diabetes and cancer (use with medical supervision).",
    "Curry Leaf": "(Murraya koenigii) - Improves digestion and helps manage diabetes.",
    "Doddapatre": "(Plectranthus amboinicus) - Treats coughs, colds, and skin infections.",
    "Mint": "(Mentha spp.) - Relieves digestive issues and freshens breath.",
    "Neem": "(Azadirachta indica) - Known for its antibacterial properties and treating skin conditions.",
    "Palak(Spinach)": "(Spinacia oleracea) - Rich in iron, boosts immunity and treats anemia.",
    "Papaya": "(Carica papaya) - Aids digestion and improves skin health.",
    "Tamarind": "(Tamarindus indica) - Used to improve digestion and as a natural laxative.",
    "Tulasi": "(Ocimum sanctum) - Known for its immune-boosting and anti-inflammatory properties."
}

# Function for image segmentation
def segment_plant_leaf(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    result = cv2.bitwise_and(image, image, mask=mask)
    rgba = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask
    rgb = cv2.cvtColor(rgba, cv2.COLOR_BGRA2BGR)
    return rgb

# Function for model prediction
def predict_image(image: np.ndarray):
    image_resized = cv2.resize(image, (256, 256))
    image_normalized = image_resized.astype('float32') / 255.0
    image_input = np.expand_dims(image_normalized, axis=0)
    prediction = model.predict(image_input)
    predicted_class_idx = np.argmax(prediction, axis=1)[0]
    predicted_class_name = class_labels[predicted_class_idx]
    confidence = float(np.max(prediction))
    return predicted_class_name, confidence

# Fetch description from local dictionary
def get_plant_description(plant_name):
    return plant_descriptions.get(plant_name, "No description available.")

# Create the prediction API endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        segmented_image = segment_plant_leaf(image)
        predicted_class, confidence = predict_image(segmented_image)
        
        # Get plant description
        plant_info = get_plant_description(predicted_class)

        return JSONResponse(content={
            "plant_name": predicted_class,
            "confidence": round(confidence, 3),
            "description": plant_info
        })

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
