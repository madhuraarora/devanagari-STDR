import os
import numpy as np
from PIL import Image, ImageFilter

os.makedirs("backgrounds", exist_ok=True)

for i in range(5):
    # random light background with noise
    arr = np.random.randint(180, 255, (32, 128), dtype=np.uint8)
    img = Image.fromarray(arr)
    
    # slight blur for realism
    if i % 2 == 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    img.save(f"backgrounds/bg{i+1}.png")

print("5 test background images saved in 'backgrounds/' folder")