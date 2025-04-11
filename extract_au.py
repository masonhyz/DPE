import os
import argparse
import subprocess
import glob

def extract_aus_from_images(directory):
    # Find all images matching the pattern
    image_paths = glob.glob(os.path.join(directory, 'video*_*_*.jpeg'))

    for image_path in image_paths:
        # Construct the output CSV filename
        base_name = os.path.basename(image_path)
        name_without_ext = os.path.splitext(base_name)[0]
        output_csv = os.path.join(directory, f"{name_without_ext}_aus.csv")

        # Run LibreFace on the image
        subprocess.run([
            'libreface',
            f'--input_path={image_path}',
            f'--output_path={output_csv}'
        ])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract AUs from images using LibreFace.")
    parser.add_argument('--dir', type=str, required=True, help='Directory containing the image pairs.')
    args = parser.parse_args()

    extract_aus_from_images(args.dir)
