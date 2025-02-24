import sqlite3
import json
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import defaultdict

# Constants
DB_PATH = "dataset.db"
IMAGE_FOLDER = "./inputs/"
IMAGE_SIZE = 2000  # Expected image size

def get_tight_bounding_box(mask):
    """
    Compute the smallest bounding box around the object mask.
    
    - mask: Full-size 2000x2000 binary mask of the object.
    
    Returns: (x, y, w, h) in full 2000x2000 image coordinates.
    """
    # Find non-zero pixels in the mask
    coords = np.column_stack(np.where(mask > 0))

    if len(coords) == 0:
        return None  # No object detected, return None

    # Get min and max bounds
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Compute width and height
    w = x_max - x_min
    h = y_max - y_min

    return (x_min, y_min, w, h)


def load_data_from_db():
    """Load all image data from the database and check for duplicates BEFORE modifying bounding boxes."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT image_index, object_name, x_coord, y_coord, width, height, mask FROM images")
    data = c.fetchall()
    conn.close()

    image_data = defaultdict(list)
    object_positions = set()  # Track positions to detect duplicates

    for entry in data:
        image_index, object_name, x, y, w, h, mask_blob = entry

        # **Check for duplicates BEFORE modifying bounding box**
        position_tuple = (image_index, object_name, x, y, w, h)
        if position_tuple in object_positions:
            print(f"❌ Duplicate Found: {object_name} at ({x}, {y}, {w}, {h}) in image {image_index}")
            continue  # Skip duplicates
        object_positions.add(position_tuple)

        # Decode mask
        mask = cv2.imdecode(np.frombuffer(mask_blob, np.uint8), cv2.IMREAD_GRAYSCALE)

        # Ensure mask is correctly placed in a full 2000x2000 frame
        full_size_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        if y + h <= IMAGE_SIZE and x + w <= IMAGE_SIZE:
            full_size_mask[y:y+h, x:x+w] = mask
        else:
            print(f"⚠️ Warning: Object {object_name} at ({x}, {y}, {w}, {h}) exceeds bounds.")

        # **Compute optimized bounding box AFTER checking duplicates**
        updated_bbox = get_tight_bounding_box(full_size_mask)
        if updated_bbox is None:
            continue  # Skip if no object is detected

        obj_x, obj_y, obj_w, obj_h = updated_bbox

        image_data[image_index].append({
            "class_name": object_name,
            "bbox": [obj_x, obj_y, obj_x + obj_w, obj_y + obj_h],  # Updated tight bounding box
            "mask": full_size_mask
        })

    return image_data


def visualize_data(image_data):
    """Visualize images with bounding boxes and masks."""
    for image_index, objects in image_data.items():
        image_path = f"{IMAGE_FOLDER}/image_{image_index}.jpg"
        
        if not os.path.exists(image_path):
            print(f"❌ Missing image: {image_path}")
            continue

        # Load the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        combined_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

        for obj in objects:
            x1, y1, x2, y2 = obj["bbox"]

            # Fix color issue by setting the correct BGR value
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Now Red
            cv2.putText(image, obj["class_name"], (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)  # Red text

            # Overlay mask
            combined_mask = np.maximum(combined_mask, obj["mask"])

        ax[0].imshow(image)  # Ensure the image is updated before displaying
        ax[0].set_title(f"Bounding Boxes - Image {image_index}")

        ax[1].imshow(combined_mask, cmap="gray")
        ax[1].set_title(f"Masks - Image {image_index}")

        plt.show()  # Show the plot


# **Run the pipeline**
print("🔍 Checking database for duplicate objects...")
image_data = load_data_from_db()
visualize_data(image_data)
print("✅ Database check completed.")
