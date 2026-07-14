from PIL import Image, ImageDraw, ImageFont
import os
import glob

OUTPUTS_DIR = os.path.expanduser("~/road_damage_project/outputs")
FONT_SIZE   = 36
TITLE_HEIGHT = 60
BG_COLOR    = (255, 255, 255)
TEXT_COLOR  = (0, 0, 0)
OVERWRITE   = True
LEFT_TITLE  = "Original Image"
RIGHT_TITLE = "Damage Assessment"

def load_font(size):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()

pattern = os.path.join(OUTPUTS_DIR, "**", "*_damage_map.png")
files   = glob.glob(pattern, recursive=True)

if not files:
    print(f"Hiç *_damage_map.png bulunamadı: {pattern}")
    exit(1)

font = load_font(FONT_SIZE)

for fpath in files:
    img = Image.open(fpath).convert("RGB")
    W, H = img.size
    panel_w = W // 2
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, 0, panel_w - 1, TITLE_HEIGHT], fill=BG_COLOR)
    cx_left = panel_w // 2
    cy      = TITLE_HEIGHT // 2
    draw.text((cx_left, cy), LEFT_TITLE, font=font, fill=TEXT_COLOR, anchor="mm")

    draw.rectangle([panel_w, 0, W - 1, TITLE_HEIGHT], fill=BG_COLOR)
    cx_right = panel_w + panel_w // 2
    draw.text((cx_right, cy), RIGHT_TITLE, font=font, fill=TEXT_COLOR, anchor="mm")

    out_path = fpath if OVERWRITE else fpath.replace(".png", "_fixed.png")
    img.save(out_path, "PNG")
    print(f"✓  {os.path.relpath(out_path, OUTPUTS_DIR)}")

print(f"\nToplam {len(files)} dosya güncellendi.")
