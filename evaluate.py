# src/evaluate.py
"""
Evaluate script using sklearn metrics.

Usage:
python src/evaluate.py --test_csv data/splits/test.csv --mlb data/splits/mlb.pkl --model models/resnet_genre.h5
"""
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from src.utils import load_mlb
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

def prepare_test_dataset(df, mlb, image_size):
    paths = df['image_path'].tolist()
    labels = mlb.transform(df['genres'])
    def _parse(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3)
        image = tf.image.resize(image, (image_size, image_size))
        image = tf.cast(image, tf.float32) / 255.0
        return image
    images = np.stack([_parse(p).numpy() for p in paths])
    return images, labels

def main(test_csv, mlb_path, model_path, image_size=224):
    mlb = load_mlb(mlb_path)
    df = pd.read_csv(test_csv)
    df['genres'] = df['genres'].apply(lambda s: eval(s) if isinstance(s, str) and s.startswith('[') else s if isinstance(s, list) else str(s).split('|'))

    model = tf.keras.models.load_model(model_path)
    X_test, y_test = prepare_test_dataset(df, mlb, image_size)
    y_pred = model.predict(X_test, batch_size=32)

    # Binarize using threshold 0.5
    y_pred_bin = (y_pred >= 0.5).astype(int)

    print("Classification report (per-genre):")
    print(classification_report(y_test, y_pred_bin, target_names=mlb.classes_, zero_division=0))

    try:
        aucs = roc_auc_score(y_test, y_pred, average=None)
        print("Per-genre AUCs:")
        for g, a in zip(mlb.classes_, aucs):
            print(f"{g}: {a:.4f}")
        print("Average AUC (macro):", np.nanmean(aucs))
    except Exception as e:
        print("AUC calculation failed:", e)

    # Average precision (PR AUC)
    try:
        aps = average_precision_score(y_test, y_pred, average=None)
        print("Per-genre Average Precision (AP):")
        for g, a in zip(mlb.classes_, aps):
            print(f"{g}: {a:.4f}")
        print("Mean AP:", np.nanmean(aps))
    except Exception as e:
        print("AP calculation failed:", e)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_csv', required=True)
    parser.add_argument('--mlb', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--image_size', type=int, default=224)
    args = parser.parse_args()
    main(args.test_csv, args.mlb, args.model, args.image_size)
