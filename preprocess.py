# src/preprocess.py
"""
Create train/val/test CSVs and fit MultiLabelBinarizer.
Run:
  python src/preprocess.py --labels_csv path/to/labels.csv --out_dir data/splits
"""
import argparse
import os
from src.utils import load_labels_csv, create_splits, fit_mlb_and_save, df_to_csv_split

def main(labels_csv, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df = load_labels_csv(labels_csv)
    train, val, test = create_splits(df)
    train_csv = os.path.join(out_dir, 'train.csv')
    val_csv = os.path.join(out_dir, 'val.csv')
    test_csv = os.path.join(out_dir, 'test.csv')
    df_to_csv_split(train, train_csv)
    df_to_csv_split(val, val_csv)
    df_to_csv_split(test, test_csv)
    print(f"Splits saved: {train_csv}, {val_csv}, {test_csv}")
    mlb = fit_mlb_and_save(train, save_path=os.path.join(out_dir, 'mlb.pkl'))
    print(f"Saved mlb to {os.path.join(out_dir, 'mlb.pkl')}")
    print("Genres:", mlb.classes_)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels_csv', required=True, help='CSV with image_path and genres (pipe-separated)')
    parser.add_argument('--out_dir', default='data/splits', help='where to save train/val/test and mlb')
    args = parser.parse_args()
    main(args.labels_csv, args.out_dir)
