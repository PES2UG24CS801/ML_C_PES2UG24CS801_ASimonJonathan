from flask import Flask, render_template, request
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn
import os
from werkzeug.utils import secure_filename
import json
import time

app = Flask(__name__)

# ------------------- MODEL SETUP -------------------
GENRES_FILE = "genres.txt"
MODEL_PATH = "movie_genre_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load genres
with open(GENRES_FILE, "r", encoding="utf8") as f:
    GENRES = [line.strip() for line in f.readlines()]

# Create model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(GENRES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Upload folder setup
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# History file setup
HISTORY_FILE = 'history.json'

# Initialize history file if it doesn't exist
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, 'w') as f:
        json.dump([], f)

# ------------------- ROUTES -------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/predict_result', methods=['POST'])
def predict_result():
    if 'poster' not in request.files:
        app.logger.error("No file part in the request")
        return render_template('predict.html', error="No file uploaded! Please try again.")
    
    file = request.files['poster']
    if file.filename == '':
        app.logger.error("No file selected")
        return render_template('predict.html', error="No file selected! Please select an image.")
    
    try:
        # Save the uploaded image
        filename = secure_filename(f"{int(time.time())}_{file.filename}")
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        app.logger.info(f"File saved: {file_path}")

        # Process image for prediction
        image = Image.open(file_path).convert('RGB')
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.sigmoid(output).squeeze(0).cpu().numpy()

        # Multi-label prediction with threshold
        threshold = 0.5
        predicted_genres = [GENRES[i] for i, p in enumerate(probs) if p >= threshold]
        if not predicted_genres:
            predicted_genres = ["None"]

        confidences = [round(float(probs[i])*100, 2) for i, p in enumerate(probs) if p >= threshold]

        # Save to history
        with open(HISTORY_FILE, 'r+') as f:
            history = json.load(f)
            history.append({
                'image_filename': filename,
                'genres': predicted_genres,
                'confidences': confidences,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            })
            f.seek(0)
            json.dump(history, f, indent=2)

        return render_template('result.html', genres=predicted_genres, confidences=confidences, image_filename=filename)
    except Exception as e:
        app.logger.error(f"Error processing image: {str(e)}")
        return render_template('predict.html', error=f"Error processing image: {str(e)}")

@app.route('/history')
def history_page():
    try:
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)
        return render_template('history.html', history=history)
    except Exception as e:
        app.logger.error(f"Error loading history: {str(e)}")
        return render_template('history.html', history=[], error="Could not load history.")

# ------------------- RUN -------------------
if __name__ == '__main__':
    app.run(debug=True)