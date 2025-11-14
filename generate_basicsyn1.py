
import uharfbuzz as hb
import freetype
from PIL import Image
import numpy as np
import os
import random


def shape_text(text, font_path, font_size=64):
    face = freetype.Face(font_path)
    face.set_char_size(font_size * 64)

    fontdata = open(font_path, "rb").read()
    hb_face = hb.Face(fontdata)
    hb_font = hb.Font(hb_face)
    hb_font.scale = (face.size.height, face.size.height)

    if hasattr(hb, "ot_font_set_funcs"):
        hb.ot_font_set_funcs(hb_font)
    elif hasattr(hb, "ft_font_set_funcs"):
        hb.ft_font_set_funcs(hb_font, face)

    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(hb_font, buf)

    infos = buf.glyph_infos
    positions = buf.glyph_positions

    glyphs = []
    for info, pos in zip(infos, positions):
        glyphs.append({
            "gid": info.codepoint,
            "x_offset": pos.x_offset / 64.0,
            "y_offset": pos.y_offset / 64.0,
            "x_advance": pos.x_advance / 64.0,
            "y_advance": pos.y_advance / 64.0
        })

    return glyphs, face


def render_text_image(text, font_path, font_size=64, margin=10):
    glyphs, face = shape_text(text, font_path, font_size)

    width = int(sum(g["x_advance"] for g in glyphs)) + 2 * margin
    height = int(font_size * 2)
    img = Image.new("L", (width, height), color=255)
    img_array = np.array(img, dtype=np.uint8)

    pen_x = margin
    pen_y = font_size + margin

    for g in glyphs:
        face.load_glyph(g["gid"], freetype.FT_LOAD_RENDER)
        bitmap = face.glyph.bitmap
        top = face.glyph.bitmap_top
        left = face.glyph.bitmap_left

        if bitmap.width == 0 or bitmap.rows == 0:
            pen_x += g["x_advance"]
            continue

        glyph_img = np.array(bitmap.buffer, dtype=np.uint8).reshape(bitmap.rows, bitmap.width)
        x = int(pen_x + left + g["x_offset"])
        y = int(pen_y - top - g["y_offset"])

        # Safe bounds
        y1, y2 = max(y, 0), min(y + bitmap.rows, img_array.shape[0])
        x1, x2 = max(x, 0), min(x + bitmap.width, img_array.shape[1])
        gy1, gy2 = max(0, y1 - y), max(0, y1 - y) + (y2 - y1)
        gx1, gx2 = max(0, x1 - x), max(0, x1 - x) + (x2 - x1)

        if y2 > y1 and x2 > x1 and gy2 > gy1 and gx2 > gx1:
            img_array[y1:y2, x1:x2] = np.minimum(
                img_array[y1:y2, x1:x2],
                255 - glyph_img[gy1:gy2, gx1:gx2]
            )

        pen_x += g["x_advance"] * 0.56

    img = Image.fromarray(img_array, mode="L")

    # small random tilt and position jitter
    angle = random.uniform(-2, 2)
    img = img.rotate(angle, expand=True, fillcolor=255)

    dx = random.randint(-3, 3)
    dy = random.randint(-3, 3)
    translated = Image.new("L", img.size, 255)
    translated.paste(img, (dx, dy))
    return translated


if __name__ == "__main__":
    charset_file = "master_charset.txt"
    fonts_dir = "dev-font"
    output_dir = "images/train_basic/basic_syn1"
    os.makedirs(output_dir, exist_ok=True)

    label_file_path = os.path.join(output_dir, "basic_syn1.txt")

    # Load all fonts
    font_paths = [os.path.join(fonts_dir, f) for f in os.listdir(fonts_dir) if f.endswith(".ttf")]
    if not font_paths:
        raise FileNotFoundError(" No .ttf fonts found in dev-font folder.")

    # Load charset
    with open(charset_file, "r", encoding="utf-8") as f:
        charset = [line.strip() for line in f if line.strip()]

    total_images = len(charset) * 80
    print(f"Generating {total_images} images ({len(charset)} chars × 80 variations each)...")

    # Open label file for writing
    with open(label_file_path, "w", encoding="utf-8") as label_file:
        for idx, char in enumerate(charset):
            for variation in range(80):
                font_path = random.choice(font_paths)
                img = render_text_image(char, font_path, font_size=72)
                filename = f"char_{idx:04d}_var_{variation:02d}.png"
                full_path = os.path.join(output_dir, filename)
                img.save(full_path)

                # Write label: filename + actual character
                label_file.write(f"{filename} {char}\n")

    print(f"Generated {total_images} images in {output_dir}")
    print(f"Labels saved to {label_file_path}")
