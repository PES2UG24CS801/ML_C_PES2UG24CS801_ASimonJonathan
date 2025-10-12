# utils.py
import os
import io
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import requests

class MovieGenreDataset(Dataset):
    """
    CSV expected columns: imdbId, Poster (optional, URL), Genre
    - imdbId used to map to local files: data/Posters/<imdbId>.jpg
    - If local not found and Poster URL provided the URL will be downloaded and cached in data/cache_posters
    - Genres are pipe-separated in the 'Genre' column, e.g. "Action|Adventure"
    """
    def __init__(self, csv_file, img_dir="data/Posters", transform=None, cache_dir="data/cache_posters"):
        # safe read (handles odd characters)
        self.data = pd.read_csv(csv_file, encoding='latin1')
        self.img_dir = img_dir
        self.transform = transform

        # create cache dir for downloaded images
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir

        # compute unique genres
        genres_all = set()
        for val in self.data['Genre'].dropna().astype(str):
            for g in val.split('|'):
                g = g.strip()
                if g:
                    genres_all.add(g)
        self.genres = sorted(list(genres_all))
        self.genre_to_idx = {g: i for i, g in enumerate(self.genres)}

        print(f"Loaded dataset CSV: {csv_file} ({len(self.data)} rows), found {len(self.genres)} genres.")

    def __len__(self):
        return len(self.data)

    def _load_local(self, imdb_id):
        path = os.path.join(self.img_dir, f"{imdb_id}.jpg")
        if os.path.exists(path):
            try:
                img = Image.open(path).convert("RGB")
                return img
            except Exception:
                return None
        return None

    def _load_cached(self, idx):
        # cached filename based on row index to avoid collision
        cache_path = os.path.join(self.cache_dir, f"{idx}.jpg")
        if os.path.exists(cache_path):
            try:
                return Image.open(cache_path).convert("RGB")
            except Exception:
                return None
        return None

    def _download_and_cache(self, url, idx):
        try:
            resp = requests.get(url, timeout=8)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            # save to cache
            cache_path = os.path.join(self.cache_dir, f"{idx}.jpg")
            img.save(cache_path)
            return img
        except Exception:
            return None

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        imdb_id = str(row['imdbId']).strip() if 'imdbId' in row else None
        poster_url = row['Poster'] if 'Poster' in row else None

        # try local imdbId.jpg first
        img = None
        if imdb_id:
            img = self._load_local(imdb_id)

        # try cache
        if img is None:
            img = self._load_cached(idx)

        # try download from Poster URL
        if img is None and isinstance(poster_url, str) and poster_url.startswith("http"):
            img = self._download_and_cache(poster_url, idx)

        # if still None -> raise so you know which rows are missing
        if img is None:
            raise FileNotFoundError(f"Image not found locally or via URL for row idx={idx}, imdbId={imdb_id}")

        if self.transform:
            img = self.transform(img)

        # build multi-hot label
        label = torch.zeros(len(self.genres), dtype=torch.float32)
        if isinstance(row['Genre'], str):
            for g in row['Genre'].split('|'):
                g = g.strip()
                if g in self.genre_to_idx:
                    label[self.genre_to_idx[g]] = 1.0

        return img, label
