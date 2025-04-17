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

# Import from LibreFace
import LibreFace.libreface as libreface

def split_image_half(image_path):
    """Split image into left and right halves."""
    img = Image.open(image_path)
    w, h = img.size
    left = img.crop((0, 0, w // 2, h))
    right = img.crop((w // 2, 0, w, h))
    return left, right

def extract_aus_from_pil(image_pil, tag):
    """Extract AU data from a PIL image."""
    try:
        result = libreface.get_facial_attributes(image_pil)
        if result:
            result["source"] = tag
            return result
        else:
            print(f"[WARNING] No facial attributes for {tag}")
            return None
    except Exception as e:
        print(f"[ERROR] Failed to process {tag}: {e}")
        return None

def main(image_dir):
    image_paths = [os.path.join(image_dir, f)
                   for f in os.listdir(image_dir)
                   if f.lower().endswith(".png")]

    all_results = []

    for img_path in tqdm(sorted(image_paths), desc="Processing Images"):
        filename = os.path.basename(img_path)
        left_img, right_img = split_image_half(img_path)

        left_result = extract_aus_from_pil(left_img, f"{filename}_left")
        right_result = extract_aus_from_pil(right_img, f"{filename}_right")

        for res in [left_result, right_result]:
            if res and res.get("au_intensities"):
                data = res["au_intensities"]
                data["filename"] = res["source"]
                all_results.append(data)

    if all_results:
        df = pd.DataFrame(all_results)
        out_path = os.path.join(image_dir, "split_faces_aus.csv")
        df.to_csv(out_path, index=False)
        print(f"[INFO] Saved split AU data to: {out_path}")
    else:
        print("[INFO] No AU data extracted from any image.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split each .png into left/right face and extract AUs.")
    parser.add_argument("--dir", type=str, required=True, help="Path to the directory containing .png images")
    args = parser.parse_args()
    main(args.dir)
