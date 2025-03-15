from ultralytics import YOLO
import cv2
import numpy as np
from random import randint
from sqlite3 import connect
from multiprocessing import Pool, Lock, cpu_count, freeze_support
from os import listdir
import random

NUM_OBJECTS_PER_IMAGE_MIN = 25
NUM_OBJECTS_PER_IMAGE_MAX = 45
NUM_IMAGES_TO_CREATE = 10000
PARALLEL_WORKERS = min(cpu_count(), 14)
DB_LOCK = Lock()

model_seg = YOLO("yolo11n-seg.pt")


def get_max_image_index():
    """Fetch the latest image index from the database."""
    with connect('dataset.db') as conn:
        c = conn.cursor()
        c.execute('SELECT MAX(image_index) FROM images')
        max_index = c.fetchone()[0]
    return max_index if max_index is not None else 0


def get_random_image():
    """Select a random image from the dataset folder."""
    folders = listdir('./dataset')
    if not folders:
        return None
    selected_folder = random.choice(folders)
    images = listdir(f'./dataset/{selected_folder}')
    if not images:
        return None
    return (selected_folder, random.choice(images))


def create_image(image_index):
    """Generate a synthetic image with segmented objects."""
    background_image = np.zeros((2000, 2000, 3), dtype=np.uint8)
    object_count = randint(NUM_OBJECTS_PER_IMAGE_MIN, NUM_OBJECTS_PER_IMAGE_MAX)

    for _ in range(object_count):
        folder, image = get_random_image()
        image_path = f"./dataset/{folder}/{image}"
        object_image = cv2.imread(image_path)
        object_image = cv2.resize(object_image, (640, 480))

        results_seg = model_seg.predict(image_path, save=False)
        h, w = 480, 640
        x_offset = randint(0, 2000 - w)
        y_offset = randint(0, 2000 - h)

        try:
            for result in results_seg:
                for mask in result.masks:
                    mask_array = mask.cpu().data.numpy()[0]
                    mask_3channel = np.stack([mask_array] * 3, axis=-1)

                    roi = background_image[y_offset:y_offset+h, x_offset:x_offset+w]
                    masked_object = object_image * mask_3channel
                    masked_background = roi * (1 - mask_3channel)
                    background_image[y_offset:y_offset+h, x_offset:x_offset+w] = masked_object + masked_background

            with DB_LOCK:
                with connect('dataset.db') as conn:
                    c = conn.cursor()
                    c.execute(
                        'INSERT INTO images (image_index, object_name, x_coord, y_coord, width, height) VALUES (?, ?, ?, ?, ?, ?)',
                        (image_index, folder, x_offset, y_offset, w, h)
                    )
                    conn.commit()

        except Exception as e:
            pass

    cv2.imwrite(f'./inputs/image_{image_index}.jpg', background_image)


def create_dataset():
    """Generate images in parallel."""
    current_images = get_max_image_index()
    image_indices = range(current_images + 1, current_images + NUM_IMAGES_TO_CREATE + 1)

    with Pool(PARALLEL_WORKERS) as pool:
        pool.map(create_image, image_indices)


if __name__ == "__main__":
    freeze_support()
    conn = connect('dataset.db')
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS images (
            image_index INTEGER PRIMARY KEY,
            object_name TEXT NOT NULL,
            x_coord INTEGER NOT NULL,
            y_coord INTEGER NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL
            
        )
    ''')

    conn.commit()
    conn.close()

    create_dataset()