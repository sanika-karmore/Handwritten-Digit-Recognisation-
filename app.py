import os
import torch
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template
from models.model import DigitRecognizer
from PIL import Image
import base64
import io

app = Flask(__name__)

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DigitRecognizer().to(device)
model.load_state_dict(torch.load('models/digit_recognizer.pth', map_location=device))
model.eval()

def preprocess_image(image_data):
    # Decode base64 image
    image_data = base64.b64decode(image_data.split(',')[1])
    image = Image.open(io.BytesIO(image_data))
    
    # Convert to grayscale
    image = image.convert('L')
    
    # Resize to 28x28
    image = image.resize((28, 28))
    
    # Convert to numpy array
    image = np.array(image)
    
    # Normalize and invert
    image = 255 - image
    image = image / 255.0
    
    # Convert to tensor
    image = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)
    return image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image data from request
        image_data = request.json['image']
        
        # Preprocess image
        image = preprocess_image(image_data)
        
        # Make prediction
        with torch.no_grad():
            output = model(image.to(device))
            prediction = output.argmax(dim=1).item()
        
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 