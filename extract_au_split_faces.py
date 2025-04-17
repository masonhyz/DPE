import os
import sys
import argparse
import tempfile
from PIL import Image
import pandas as pd
from tqdm import tqdm

# Add LibreFace to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
libreface_path = os.path.join(current_dir, "LibreFace")
sys.path.append(libreface_path)

# Import from LibreFace
import LibreFace.libreface as libreface

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

def extract_aus_from_image(image_path, tag=None):
    try:
        result = libreface.get_facial_attributes(image_path)
        if result:
            result["source"] = tag if tag else os.path.basename(image_path)
            return result
        else:
            print(f"[WARNING] No facial attributes for {image_path}")
            return None
    except Exception as e:
        print(f"[ERROR] Failed to process {image_path}: {e}")
        return None

def main(image_dir):
    image_paths = [os.path.join(image_dir, f)
                   for f in os.listdir(image_dir)
                   if f.lower().endswith(".jpg")]

    all_results = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for img_path in tqdm(sorted(image_paths), desc="Processing Images"):
            filename = os.path.basename(img_path)
            left_path, right_path = split_and_save_temp(img_path, temp_dir)

            left_result = extract_aus_from_image(left_path, f"{filename}_left")
            right_result = extract_aus_from_image(right_path, f"{filename}_right")

            for res in [left_result, right_result]:
                if res:
                    flat_result = res.copy()

                    # Flatten AU and detection data
                    detected_aus = flat_result.pop("detected_aus", {})
                    au_intensities = flat_result.pop("au_intensities", {})
                    flat_keys = ["facial_expression", "pitch", "yaw", "roll"]
                    flat_result = {k: flat_result.get(k, None) for k in flat_keys}
                    ordered_result = {
                        "filename": res.get("source", "unknown"),
                        **detected_aus,
                        **au_intensities,
                        **flat_result  # everything else (landmarks, expression, pose, etc.)
                    }

                    all_results.append(ordered_result)
                    print(ordered_result)
        

    if all_results:
        df = pd.DataFrame(all_results)
        out_path = os.path.join(image_dir, "split_faces_aus.csv")
        df.to_csv(out_path, index=False)
        print(f"[INFO] Saved split AU data to: {out_path}")
    else:
        print("[INFO] No AU data extracted from any image.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split each .jpg into left/right face and extract AUs.")
    parser.add_argument("--dir", type=str, required=True, help="Path to the directory containing .jpg images")
    args = parser.parse_args()
    main(args.dir)
