import os
import sys
import argparse
import tempfile
import pandas as pd
from tqdm import tqdm
from PIL import Image

# Import py-feat core interface
from feat import Detector

# Initialize the face detector
detector = Detector()

def split_and_save_temp(image_path, temp_dir):
    """Split image and save left/right as temp files. Returns two temp paths."""
    img = Image.open(image_path)
    w, h = img.size
    left = img.crop((0, 0, w // 2, h))
    right = img.crop((w // 2, 0, w, h))

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    left_path = os.path.join(temp_dir, f"{base_name}_left.jpg")
    right_path = os.path.join(temp_dir, f"{base_name}_right.jpg")

    left.save(left_path)
    right.save(right_path)

    return left_path, right_path

def extract_feat_features(image_path, tag=None):
    # try:
    df = detector.detect_image(image_path)
    if df is not None and not df.empty:
        df["filename"] = tag if tag else os.path.basename(image_path)
        return df
    else:
        print(f"[WARNING] No face detected in: {image_path}")
        return None
    # except Exception as e:
    #     print(f"[ERROR] Failed to process {image_path}: {e}")
    #     return None

def main(image_dir):
    image_paths = [os.path.join(image_dir, f)
                   for f in os.listdir(image_dir)
                   if f.lower().endswith(".jpg")]

    all_dfs = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for img_path in tqdm(sorted(image_paths), desc="Processing Images"):
            filename = os.path.basename(img_path)
            left_path, right_path = split_and_save_temp(img_path, temp_dir)

            for face_path, suffix in zip([left_path, right_path], ["_left", "_right"]):
                df = extract_feat_features(face_path, f"{filename}{suffix}")
                if df is not None:
                    all_dfs.append(df)

    if all_dfs:
        full_df = pd.concat(all_dfs, ignore_index=True)
        out_path = os.path.join(image_dir, "pyfeat_all_features.csv")
        full_df.to_csv(out_path, index=False)
        print(f"[INFO] Saved py-feat data to: {out_path}")
    else:
        print("[INFO] No data extracted from any image.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split .jpg into left/right face and extract py-feat features.")
    parser.add_argument("--dir", type=str, required=True, help="Path to the directory containing .jpg images")
    args = parser.parse_args()
    main(args.dir)
