import os
from PIL import Image
from random import uniform

INPUT_FOLDER = "objects"
OUTPUT_FOLDER = "processed_objects"
TARGET_SIZE = (224, 224)
BACKGROUND_COLOR = (128, 128, 128)
ROTATION_ANGLE = 90

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def preprocess_image(image_path, output_path, target_size=TARGET_SIZE, background_color=BACKGROUND_COLOR, apply_rotation=True):
    """Preprocess an image for ResNet-34 input:
       - Resize if needed
       - Otherwise, paste on a gray background
    """
    img = Image.open(image_path).convert("RGB")  # resnet needs rgb

    # apply some random rotation to the image before processing
    if apply_rotation:
        angle = uniform(-ROTATION_ANGLE, ROTATION_ANGLE)
        img = img.rotate(angle, resample=Image.BICUBIC, expand=True)

    # resize normally if both dimensions are >= target size
    if img.size[0] >= target_size[0] and img.size[1] >= target_size[1]:
        img = img.resize(target_size, Image.LANCZOS)
    else:
        # create a new gray background
        bg = Image.new("RGB", target_size, background_color)
        
        # resize while keeping aspect ratio
        img.thumbnail(target_size, Image.LANCZOS)

        # center the image
        x_offset = (target_size[0] - img.size[0]) // 2
        y_offset = (target_size[1] - img.size[1]) // 2
        bg.paste(img, (x_offset, y_offset))

        img = bg

    img.save(output_path)

image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('png', 'jpg', 'jpeg', 'webp'))]

for img_file in image_files:
    input_path = os.path.join(INPUT_FOLDER, img_file)
    output_path = os.path.join(OUTPUT_FOLDER, img_file)
    preprocess_image(input_path, output_path)

print(f"Processing complete! All processed objects saved in '{OUTPUT_FOLDER}'.")