import glob
import argparse
import pandas as pd
from tqdm import tqdm

import sys
import os

# Automatically add LibreFace to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
libreface_path = os.path.join(current_dir, "LibreFace")
sys.path.append(libreface_path)

# Now you can import
from libreface.detector import Detector
from libreface.config import Config


def extract_aus(image_path, detector):
    # Get facial info
    pred = detector.detect(image_path)

    # Check that prediction succeeded
    if pred is None or pred['aus'] is None:
        print(f"[WARNING] No AUs found in: {image_path}")
        return None

    # Return AU values as dictionary
    return pred['aus']

def save_aus_to_csv(au_dict, save_path):
    if au_dict is None:
        return
    df = pd.DataFrame([au_dict])
    df.to_csv(save_path, index=False)

def main(image_dir):
    # Load detector
    config = Config()
    config.detector.face_detector.model_name = "yunet"
    detector = Detector(config)

    # Find all matching images
    images = glob.glob(os.path.join(image_dir, "video*_*_*.jpeg"))

    for img_path in tqdm(sorted(images), desc="Extracting AUs"):
        base = os.path.basename(img_path)
        name_no_ext = os.path.splitext(base)[0]
        out_path = os.path.join(image_dir, f"{name_no_ext}_aus.csv")

        aus = extract_aus(img_path, detector)
        save_aus_to_csv(aus, out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract AUs using LibreFace.")
    parser.add_argument("--dir", type=str, required=True, help="Path to directory with image pairs.")
    args = parser.parse_args()

    main(args.dir)
