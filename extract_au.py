import os
import sys
import argparse
from PIL import Image
import pandas as pd
from tqdm import tqdm

# Add LibreFace to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
libreface_path = os.path.join(current_dir, "LibreFace")
sys.path.append(libreface_path)

# Import from libreface
import LibreFace.libreface as libreface

def extract_aus_from_image(image_path):
    try:
        result = libreface.get_facial_attributes(image_path)
        print(result)
        if result and result.get("au_intensities"):
            data = result["au_intensities"]
            data["filename"] = os.path.basename(image_path)
            return data
        else:
            print(f"[WARNING] No AUs detected in {image_path}")
            return None
    except Exception as e:
        print(f"[ERROR] Failed to process {image_path}: {e}")
        return None

def main(image_dir):
    image_paths = [os.path.join(image_dir, f)
                   for f in os.listdir(image_dir)
                   if f.lower().endswith(".png")]

    all_aus = []

    for img_path in tqdm(sorted(image_paths), desc="Extracting AUs"):
        aus = extract_aus_from_image(img_path)
        if aus:
            all_aus.append(aus)

    if all_aus:
        df = pd.DataFrame(all_aus)
        out_path = os.path.join(image_dir, "extracted_aus.csv")
        df.to_csv(out_path, index=False)
        print(f"[INFO] Saved AU data to: {out_path}")
    else:
        print("[INFO] No AUs were extracted.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract AUs from .png images in a directory.")
    parser.add_argument("--dir", type=str, required=True, help="Path to the directory containing .png images")
    args = parser.parse_args()
    main(args.dir)
