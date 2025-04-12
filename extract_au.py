import os
import sys
from PIL import Image
import pandas as pd

# Add LibreFace to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
libreface_path = os.path.join(current_dir, "LibreFace")
sys.path.append(libreface_path)

# Import directly from AU_Detection
from libreface.AU_Detection.solver_inference_image import Detector

# Initialize detector
detector = Detector()

# Path to your image
image_path = "res/whole_dir3/qualified/video85_6_source.png"

# Run AU detection
results = detector.detect(image_path)

# Print AU results
if results is not None and results['aus'] is not None:
    print("Extracted AUs:")
    for k, v in results['aus'].items():
        print(f"{k}: {v:.3f}")
    
    # Save to CSV (optional)
    pd.DataFrame([results['aus']]).to_csv("your_image_aus.csv", index=False)
else:
    print("No AUs detected.")
