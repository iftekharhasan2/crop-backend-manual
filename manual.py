import os
import io
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
from tensorflow import keras
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Load ML Model
model = keras.models.load_model('tomato_disease.h5')

class_labels = [
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
]

disease_prevention = {
    "Tomato_Bacterial_spot": [
        "Prevent bacterial spot by using disease-free seeds.",
        "Implement crop rotation to reduce the disease's prevalence.",
        "Apply copper-based fungicides to control the disease."
    ],
    "Tomato_Early_blight": [
        "Prevent early blight by practicing good garden hygiene.",
        "Ensure proper watering to avoid splashing soil onto the leaves.",
        "Apply fungicides as needed to control the disease."
    ],
    "Tomato_Late_blight": [
        "Prevent late blight by providing good air circulation in your garden or greenhouse.",
        "Avoid overhead watering, as wet leaves can encourage the disease.",
        "Apply fungicides when necessary to manage the disease."
    ],
    "Tomato_Leaf_Mold": [
        "Prevent leaf mold by ensuring good air circulation and spacing between plants.",
        "Avoid wetting the leaves when watering, and water the soil instead.",
        "Apply fungicides if the disease is present and worsening."
    ],
    "Tomato_Septoria_leaf_spot": [
        "Prevent Septoria leaf spot by maintaining good garden hygiene.",
        "Avoid overhead watering to keep the leaves dry.",
        "Apply fungicides if the disease becomes a problem."
    ],
    "Tomato_Spider_mites_Two_spotted_spider_mite": [
        "Inspect plants regularly for signs of infestation.",
        "Increase humidity to discourage mites.",
        "Use insecticidal soap or neem oil to control mites."
    ],
    "Tomato__Target_Spot": [
        "Ensure good air circulation and avoid overcrowding of plants.",
        "Water at the base to keep leaves dry.",
        "Apply fungicides as needed."
    ],
    "Tomato__Tomato_YellowLeaf__Curl_Virus": [
        "Use virus-free tomato plants.",
        "Control whiteflies with insecticides.",
        "Remove and destroy infected plants."
    ],
    "Tomato__Tomato_mosaic_virus": [
        "Use virus-free seeds and disease-resistant varieties.",
        "Control aphids with insecticides.",
        "Remove and destroy infected plants."
    ],
    "Tomato_healthy": [
        "Continue monitoring for pests and diseases.",
        "Practice proper watering and fertilization."
    ]
}

def read_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).resize((256, 256))
    img = img.convert("RGB")  # Ensure 3 channels
    img = np.array(img) / 255.0  # Normalize if required by model
    return img

@app.route("/")
def home():
    return "🍅 Tomato Disease Classifier API is running."

@app.route("/api/detect", methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        image = read_image(file.read())
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        prediction = model.predict(image)
        pred_index = np.argmax(prediction)
        confidence = float(np.max(prediction[0]))
        predicted_class = class_labels[pred_index]
        prevention = disease_prevention.get(predicted_class, ['Prevention info not available.'])
        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence,
            'prevention_measures': prevention
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
