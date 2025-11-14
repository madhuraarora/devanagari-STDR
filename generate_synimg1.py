import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import torchvision.transforms as T


# Config

output_folder = "train_syn1"
os.makedirs(output_folder, exist_ok=True)

labels_file = os.path.join(output_folder, "train_syn1.txt")

vocab_file = "dev_vocab1.txt"  # 11k words

fonts_folder = "dev-font"
fonts = [os.path.join(fonts_folder, f) for f in os.listdir(fonts_folder) if f.endswith(".TTF")]

bg_folder = "backgrounds"
backgrounds = [os.path.join(bg_folder, f) for f in os.listdir(bg_folder)
               if f.lower().endswith((".jpg", ".png"))]

img_width, img_height = 128, 32
images_per_word = 25


# Augmentations

augment = T.Compose([
    T.RandomRotation(degrees=10),
    T.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=5, scale=(0.9,1.1)),
    T.RandomPerspective(distortion_scale=0.3, p=0.5),
    T.ColorJitter(brightness=0.3, contrast=0.3)
])


# Helper: Salt & pepper noise

def add_salt_pepper_noise(img, amount=0.02):
    arr = np.array(img)
    n_salt = np.ceil(amount * arr.size * 0.5).astype(int)
    n_pepper = np.ceil(amount * arr.size * 0.5).astype(int)

    # Salt
    coords = [np.random.randint(0, i, n_salt) for i in arr.shape]
    arr[coords[0], coords[1]] = 255

    # Pepper
    coords = [np.random.randint(0, i, n_pepper) for i in arr.shape]
    arr[coords[0], coords[1]] = 0

    return Image.fromarray(arr)


# Load words

with open(vocab_file, "r", encoding="utf-8") as f:
    words = [line.strip() for line in f if line.strip()]


# Generate images

with open(labels_file, "w", encoding="utf-8") as f_labels:
    idx = 0
    for word in words:
        for _ in range(images_per_word):
            # Random font and size
            font_path = random.choice(fonts)
            font_size = random.randint(20, 32)
            font = ImageFont.truetype(font_path, font_size)

            # Random background
            if backgrounds:
                bg_path = random.choice(backgrounds)
                img = Image.open(bg_path).convert("L").resize((img_width, img_height))
            else:
                img = Image.new("L", (img_width, img_height), color=255)

            draw = ImageDraw.Draw(img)

            # Random position
            bbox = draw.textbbox((0, 0), word, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = random.randint(0, max(0, img_width - text_width))
            y = random.randint(0, max(0, img_height - text_height))

            # Draw text
            draw.text((x, y), word, font=font, fill=random.randint(0, 50))

            # Optional blur
            if random.random() < 0.3:
                img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

            # Optional salt & pepper noise
            if random.random() < 0.3:
                img = add_salt_pepper_noise(img, amount=0.01)

            # Torchvision augmentations
            img = augment(img.convert("RGB")).convert("L")

            # Save image
            img_name = f"{idx}.jpg"
            img.save(os.path.join(output_folder, img_name))

            # Save label
            f_labels.write(f"{img_name} {word}\n")
            idx += 1

print(f"Generated {idx} synthetic images in '{output_folder}' with labels in '{labels_file}'")
