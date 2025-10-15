# train.py
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# -------------------
# CONFIG
# -------------------
CSV_FILE = "MovieGenre.csv"
IMG_DIR = "data/Posters"
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------
# DATASET
# -------------------
class MovieGenreDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file, encoding='latin')
        self.img_dir = img_dir
        self.transform = transform

        # Handle missing and split genres
        self.data['Genre'] = self.data['Genre'].fillna("").apply(lambda x: x.split('|'))
        all_genres = sorted(set(g for sublist in self.data['Genre'] for g in sublist if g))
        self.genres = all_genres

        # Map genre to index
        self.genre_to_idx = {g: i for i, g in enumerate(self.genres)}

        # Filter for existing images only
        self.data['filename'] = self.data['imdbId'].astype(str) + ".jpg"
        self.data = self.data[self.data['filename'].apply(lambda x: os.path.exists(os.path.join(img_dir, x)))].reset_index(drop=True)
        if len(self.data) == 0:
            raise ValueError("No usable images found! Check your CSV and image paths.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Multi-label one-hot encoding
        label = torch.zeros(len(self.genres), dtype=torch.float32)
        for g in row['Genre']:
            if g in self.genre_to_idx:
                label[self.genre_to_idx[g]] = 1.0

        return image, label

# -------------------
# TRANSFORMS
# -------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------------------
# LOAD DATA
# -------------------
dataset = MovieGenreDataset(CSV_FILE, IMG_DIR, transform=transform)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"Starting training on {len(dataset)} samples with {len(dataset.genres)} genres...")

# -------------------
# MODEL
# -------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(dataset.genres))
model = model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# -------------------
# TRAINING LOOP
# -------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    all_labels = []
    all_preds = []

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE).float()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)

        # Convert outputs to binary predictions
        preds = torch.sigmoid(outputs).detach().cpu().numpy()
        preds = (preds > 0.5).astype(int)
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

    # Combine all predictions and labels
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='micro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='micro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)

    print(f"Epoch [{epoch+1}/{EPOCHS}] | "
          f"Loss: {total_loss/len(dataset):.4f} | "
          f"Acc: {acc:.4f} | Prec: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

# -------------------
# SAVE MODEL & GENRES
# -------------------
torch.save(model.state_dict(), "movie_genre_model.pth")
with open("genres.txt", "w", encoding="utf8") as f:
    for g in dataset.genres:
        f.write(g + "\n")

print("\n✅ Training complete! Model and genres saved successfully.")
print(f"📊 Final Metrics -> Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
