from ultralytics import YOLO
import cv2
import numpy as np
import json
from random import randint
from sqlite3 import connect

# Constants
NUM_OBJECTS_PER_IMAGE_MIN = 25
NUM_OBJECTS_PER_IMAGE_MAX = 45
IMAGE_SIZE = 2000  # Background size
NUM_IMAGES_TO_CREATE = 500

# Load segmentation model
model_seg = YOLO("yolo11n-seg.pt")

# List of already placed objects and their masks
existing_objects = []

def get_max_image_index():
    """Retrieve the highest image index from the database."""
    conn = connect('dataset.db')
    c = conn.cursor()
    c.execute('SELECT MAX(image_index) FROM images')
    max_index = c.fetchone()[0]
    conn.close()
    return max_index if max_index is not None else 0

def get_random_image():
    """Select a random object image from the dataset folder."""
    from os import listdir
    import random

    folders = listdir('./dataset')
    if not folders:
        return None

    selected_folder = random.choice(folders)
    images = listdir(f'./dataset/{selected_folder}')
    if not images:
        return None

    return (selected_folder, random.choice(images))


def check_and_remove_covered_objects(temp_objects, new_object):
    """
    Modify objects' masks if they are partially covered and remove fully covered objects.
    
    - `temp_objects`: List of existing objects.
    - `new_object`: (image_index, folder, x, y, w, h, mask_array) of the new object.
    
    Returns: Updated list of objects with adjusted masks.
    """
    image_index, folder, new_x, new_y, new_w, new_h, new_mask = new_object
    final_objects = []

    for i, obj in enumerate(temp_objects):
        obj_index, obj_name, obj_x, obj_y, obj_w, obj_h, obj_mask, image = obj

        # Compute bounding box overlap
        x_overlap_start = max(obj_x, new_x)
        y_overlap_start = max(obj_y, new_y)
        x_overlap_end = min(obj_x + obj_w, new_x + new_w)
        y_overlap_end = min(obj_y + obj_h, new_y + new_h)

        if x_overlap_start < x_overlap_end and y_overlap_start < y_overlap_end:
            # Compute mask overlap region
            mask_overlap_x = x_overlap_start - obj_x
            mask_overlap_y = y_overlap_start - obj_y
            mask_overlap_w = x_overlap_end - x_overlap_start
            mask_overlap_h = y_overlap_end - y_overlap_start

            new_mask_x = x_overlap_start - new_x
            new_mask_y = y_overlap_start - new_y

            # Create a copy of the original mask before modifying
            modified_obj_mask = obj_mask.copy()

            obj_mask_region = modified_obj_mask[mask_overlap_y:mask_overlap_y + mask_overlap_h,
                                                mask_overlap_x:mask_overlap_x + mask_overlap_w]

            new_mask_region = new_mask[new_mask_y:new_mask_y + mask_overlap_h,
                                       new_mask_x:new_mask_x + mask_overlap_w]

            # Modify the object mask: Set 0 only where the new mask is 1
            obj_mask_region = np.where(new_mask_region == 1, 0, obj_mask_region)

            # **Update Bounding Box** using the modified mask
          
           
            final_objects.append((obj_index, obj_name, obj_x, obj_y, obj_w, obj_h, modified_obj_mask, image))
        else:
            # Keep objects that are not overlapping
            final_objects.append(obj)


    return final_objects


def store_final_objects_in_database(image_index):
    """Store only the final visible objects in the database."""
    conn = connect('dataset.db')
    c = conn.cursor()

    for obj in existing_objects:
        obj_index, object_name, x, y, w, h, mask, object_image = obj

        # Convert mask to binary format
        mask_binary = cv2.imencode('.png', mask)[1].tobytes()

        # Convert mask into a list of visible pixel coordinates
        pixels = [(x + i, y + j) for i in range(h) for j in range(w) if mask[i, j] > 0]
        pixel_mask_json = json.dumps(pixels)

        # Insert into database
        c.execute('''INSERT INTO images (image_index, object_name, x_coord, y_coord, width, height, mask, pixel_mask)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                  (image_index, object_name, x, y, w, h, mask_binary, pixel_mask_json))

    conn.commit()
    conn.close()


def create_image(image_index):
    """Generate a synthetic image with multiple objects."""
    global existing_objects
    background_image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)

    temp_objects = []  # Temporary list for collecting objects before applying them

    for _ in range(randint(NUM_OBJECTS_PER_IMAGE_MIN, NUM_OBJECTS_PER_IMAGE_MAX)):
        folder, image = get_random_image()
        image_path = f"./dataset/{folder}/{image}"
        object_image = cv2.imread(image_path)
        object_image = cv2.resize(object_image, (640, 480))

        results_seg = model_seg.predict(image_path, save=False)

        h, w = 480, 640
        x_offset = randint(0, IMAGE_SIZE - w)
        y_offset = randint(0, IMAGE_SIZE - h)

        try:
            for result in results_seg:

                # Select only the first valid mask
                mask = result.masks[0]  # Pick the first mask
                mask_array = mask.cpu().data.numpy()[0]  # Convert to numpy array

                # **Create a full-size (2000x2000) mask with zeros**
            
                temp_objects = check_and_remove_covered_objects(temp_objects, 
                        (image_index, folder, x_offset, y_offset, w, h, mask_array)
                    )
                print(f"✅ Processed mask for image {image_index}, placed at ({x_offset}, {y_offset})")

                # **Store each detected object separately with full-sized mask**
                temp_objects.append((image_index, folder, x_offset, y_offset, w, h, mask_array, object_image))

        except Exception as e:
            print(f"⚠️ Error processing image {image_index}: {e}")





    # Now process all final objects at once
    existing_objects = temp_objects  # Assign all collected objects to the global list
  

    # Apply objects to the background now
    for obj in existing_objects:
        _, _, x, y, w, h, mask_array, object_image= obj
        mask_3channel = np.stack([mask_array] * 3, axis=-1)

        # Use the mask to blend the object with the background
        roi = background_image[y:y+h, x:x+w]
        masked_object = object_image * mask_3channel
        masked_background = roi * (1 - mask_3channel)
        background_image[y:y+h, x:x+w] = masked_object + masked_background

    # Store only the final, visible objects in the database
    store_final_objects_in_database(image_index)

    # Save the final synthetic image
    cv2.imwrite(f'./inputs/image_{image_index}.jpg', background_image)

def create_dataset():
    """Generate multiple synthetic images for training."""
    current_images = get_max_image_index()
    for image_index in range(current_images + 1, current_images + NUM_IMAGES_TO_CREATE + 1):
        create_image(image_index)
