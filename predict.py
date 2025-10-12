# predict.py
import sys
import torch
from torchvision import models, transforms
from PIL import Image

IMG_PATH = sys.argv[1]  # pass image path as argument
MODEL_PATH = "movie_genre_model.pth"
GENRES_FILE = "genres.txt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------
# LOAD GENRES
# -------------------
with open(GENRES_FILE, "r", encoding="utf8") as f:
    genres = [line.strip() for line in f.readlines()]

# -------------------
# MODEL
# -------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, len(genres))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# -------------------
# IMAGE TRANSFORM
# -------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------------------
# PREDICT
# -------------------
image = Image.open(IMG_PATH).convert("RGB")
image = transform(image).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    output = model(image)
    probs = torch.sigmoid(output).squeeze().cpu().numpy()

# Multi-label prediction with threshold 0.5
predicted = [genres[i] for i, p in enumerate(probs) if p > 0.5]

print(f"Predicted genres for {IMG_PATH}: {predicted if predicted else 'None'}")
