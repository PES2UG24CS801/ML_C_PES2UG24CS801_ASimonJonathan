import pandas as pd
import os

csv_file = "MovieGenre.csv"
img_dir = "data/Posters"

# Use latin1 encoding to handle weird characters
df = pd.read_csv(csv_file, dtype={0: str}, encoding='latin1')  

existing_files = set(f.split(".")[0] for f in os.listdir(img_dir))  # remove .jpg
df = df[df.iloc[:, 0].isin(existing_files)]  # keep only rows with existing images

print(f"Filtered dataset size: {len(df)}")
df.to_csv("MovieGenre_filtered.csv", index=False, encoding='latin1')
