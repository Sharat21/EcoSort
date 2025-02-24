from ultralytics import YOLO
import cv2
import numpy as np
from random import randint, uniform
from sqlite3 import connect


NUM_OBJECTS_PER_IMAGE_MIN = 25
NUM_OBJECTS_PER_IMAGE_MAX = 45
RESIZE_MIN = 1.75
RESIZE_MAX = 2.25
NUM_IMAGES_TO_CREATE = 1

model_seg = YOLO("yolo11n-seg.pt")


def get_max_image_index():
    conn = connect('dataset.db')
    c = conn.cursor()
    c.execute('SELECT MAX(image_index) FROM images')
    max_index = c.fetchone()[0]
    conn.close()

    return max_index if max_index is not None else 0


def get_random_image():
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


def create_image(image_index):
    background_image = np.zeros((2000, 2000, 3), dtype=np.uint8)
    for label in range(randint(NUM_OBJECTS_PER_IMAGE_MIN, NUM_OBJECTS_PER_IMAGE_MAX)):
        folder, image = get_random_image()
        image = f"./dataset/{folder}/{image}"
        object_image = cv2.imread(image)
        object_image = cv2.resize(object_image, (640, 480))



        results_seg = model_seg.predict(image, save=False)

        h, w = 480, 640
        x_offset = randint(0, 2000 - w)
        y_offset = randint(0, 2000 - h)

        try:
            for result in results_seg:
                for mask in result.masks:
                    mask_array = mask.cpu().data.numpy()[0]  # Convert to numpy array
                    # Expand mask to 3 channels
                    mask_3channel = np.stack([mask_array] * 3, axis=-1)

                    # Use the mask to blend the object with the background
                    roi = background_image[y_offset:y_offset+h, x_offset:x_offset+w]
                    masked_object = object_image * mask_3channel
                    masked_background = roi * (1 - mask_3channel)
                    background_image[y_offset:y_offset+h, x_offset:x_offset+w] = masked_object + masked_background



            conn = connect('dataset.db')
            c = conn.cursor()
            c.execute('INSERT INTO images (image_index, object_name, x_coord, y_coord, width, height) VALUES (?, ?, ?, ?, ?, ?)', (image_index, folder, x_offset, y_offset, w, h))
            conn.commit()
            conn.close()
        except Exception as e:
            pass

    cv2.imwrite(f'./inputs/image_{image_index}.jpg', background_image)

def create_dataset():
    current_images = get_max_image_index()
    for image_index in range(current_images + 1, current_images + NUM_IMAGES_TO_CREATE + 1):
        create_image(image_index)
