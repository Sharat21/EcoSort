from multiprocessing import Pool, Queue, Process, cpu_count
from ultralytics import YOLO
import cv2
import numpy as np
import json
import os
from random import randint
from sqlite3 import connect

# Constants
NUM_OBJECTS_PER_IMAGE_MIN = 25
NUM_OBJECTS_PER_IMAGE_MAX = 45
IMAGE_SIZE = 2000
NUM_IMAGES_TO_CREATE = 1000
NUM_WORKERS = cpu_count() - 4

model_seg = YOLO("yolo11n-seg.pt")

def get_max_image_index():
    '''
    This function returns the maximum image index in the database.
    '''
    conn = connect('dataset.db')
    c = conn.cursor()
    c.execute('SELECT MAX(image_index) FROM images')
    max_index = c.fetchone()[0]
    conn.close()
    return max_index if max_index is not None else 0

def get_random_image():
    '''
    This function returns a random image from the dataset.
    '''
    folders = os.listdir('./dataset')
    if not folders:
        return None
    selected_folder = np.random.choice(folders)
    images = os.listdir(f'./dataset/{selected_folder}')
    if not images:
        return None
    return selected_folder, np.random.choice(images)

def check_and_remove_covered_objects(temp_objects, new_object):
    '''
    This function checks if the new object is covered by any of the existing objects.
    If it is, it removes the existing object.
    '''
    image_index, folder, new_x, new_y, new_w, new_h, new_mask = new_object
    final_objects = []

    for obj in temp_objects:
        obj_index, obj_name, obj_x, obj_y, obj_w, obj_h, obj_mask, image = obj

        x_overlap_start = max(obj_x, new_x)
        y_overlap_start = max(obj_y, new_y)
        x_overlap_end = min(obj_x + obj_w, new_x + new_w)
        y_overlap_end = min(obj_y + obj_h, new_y + new_h)

        if x_overlap_start < x_overlap_end and y_overlap_start < y_overlap_end:
            mask_overlap_x = x_overlap_start - obj_x
            mask_overlap_y = y_overlap_start - obj_y
            mask_overlap_w = x_overlap_end - x_overlap_start
            mask_overlap_h = y_overlap_end - y_overlap_start

            new_mask_x = x_overlap_start - new_x
            new_mask_y = y_overlap_start - new_y

            modified_obj_mask = obj_mask.copy()

            obj_mask_region = modified_obj_mask[mask_overlap_y:mask_overlap_y + mask_overlap_h,
                                                mask_overlap_x:mask_overlap_x + mask_overlap_w]

            new_mask_region = new_mask[new_mask_y:new_mask_y + mask_overlap_h,
                                       new_mask_x:new_mask_x + mask_overlap_w]

            obj_mask_region = np.where(new_mask_region == 1, 0, obj_mask_region)

            final_objects.append((obj_index, obj_name, obj_x, obj_y, obj_w, obj_h, modified_obj_mask, image))
        else:
            final_objects.append(obj)
    return final_objects

def store_final_objects_in_database(image_index, objects):
    '''
    This function stores the final objects in the database.
    '''
    conn = connect('dataset.db')
    c = conn.cursor()

    for obj in objects:
        obj_index, object_name, x, y, w, h, mask, object_image = obj
        mask_binary = cv2.imencode('.png', mask)[1].tobytes()
        pixels = [(x + i, y + j) for i in range(h) for j in range(w) if mask[i, j] > 0]
        pixel_mask_json = json.dumps(pixels)

        c.execute('''INSERT INTO images (image_index, object_name, x_coord, y_coord, width, height, mask, pixel_mask)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                  (image_index, object_name, x, y, w, h, mask_binary, pixel_mask_json))

    conn.commit()
    conn.close()

def rotate_image(image, angle):
    '''
    This function rotates an image by a given angle.
    '''
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    return rotated

def overlay_transparent(background, overlay, x, y):
    '''
    This function overlays an image on top of a background image.
    '''
    bh, bw = background.shape[:2]
    h, w = overlay.shape[:2]

    if x + w > bw or y + h > bh:
        return background

    overlay_img = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background_crop = background[y:y+h, x:x+w]
    background[y:y+h, x:x+w] = (1 - mask) * background_crop + mask * overlay_img
    return background


def add_realism(background_image):
    '''
    This function adds realism to the background image.
    '''
    grime_paths = [f"assets/grime/{f}" for f in os.listdir("assets/grime") if f.endswith(".png")]
    for _ in range(randint(3, 8)):
        grime_path = np.random.choice(grime_paths)
        grime = cv2.imread(grime_path, cv2.IMREAD_UNCHANGED)

        scale = np.random.uniform(0.3, 1.0)
        grime = cv2.resize(grime, (0, 0), fx=scale, fy=scale)
        grime = rotate_image(grime, randint(-180, 180))


        x = randint(0, IMAGE_SIZE - grime.shape[1])
        y = randint(0, IMAGE_SIZE - grime.shape[0])

        background_image = overlay_transparent(background_image, grime, x, y)
    return background_image


def create_image_worker(image_index):
    '''
    This function creates an image with a random number of objects.
    '''
    background_image = cv2.imread("assets/conveyor.jpg")
    background_image = cv2.resize(background_image, (IMAGE_SIZE, IMAGE_SIZE))
    background_image = add_realism(background_image)

    temp_objects = []

    for _ in range(randint(NUM_OBJECTS_PER_IMAGE_MIN, NUM_OBJECTS_PER_IMAGE_MAX)):
        result = get_random_image()
        if not result:
            continue
        folder, image = result
        image_path = f"./dataset/{folder}/{image}"
        object_image = cv2.imread(image_path)
        object_image = cv2.resize(object_image, (640, 480))

        object_image = rotate_image(object_image, randint(-180, 180))

        results_seg = model_seg.predict(object_image, save=False)
        h, w = 480, 640
        x_offset = randint(0, IMAGE_SIZE - w)
        y_offset = randint(0, IMAGE_SIZE - h)

        try:
            for result in results_seg:
                mask = result.masks[0]
                mask_array = mask.cpu().data.numpy()[0]

                temp_objects = check_and_remove_covered_objects(temp_objects,
                    (image_index, folder, x_offset, y_offset, w, h, mask_array))

                temp_objects.append((image_index, folder, x_offset, y_offset, w, h, mask_array, object_image))
        except Exception as e:
            print(f"Error in image {image_index}: {e}")

    for obj in temp_objects:
        _, _, x, y, w, h, mask_array, object_image = obj
        mask_3channel = np.stack([mask_array] * 3, axis=-1)
        roi = background_image[y:y+h, x:x+w]
        masked_object = object_image * mask_3channel
        masked_background = roi * (1 - mask_3channel)
        background_image[y:y+h, x:x+w] = masked_object + masked_background

    # Save image
    output_path = f'./inputs/image_{image_index}.jpg'
    cv2.imwrite(output_path, background_image)

    return (image_index, temp_objects)

def save_results_worker(queue):
    '''
    This function saves the final objects in the database.
    '''
    while True:
        result = queue.get()
        if result == 'STOP':
            break
        image_index, objects = result
        store_final_objects_in_database(image_index, objects)

def create_dataset():
    '''
    This function creates a dataset of images with a random number of objects.
    '''
    current_index = get_max_image_index()
    start_index = current_index + 1
    end_index = current_index + NUM_IMAGES_TO_CREATE + 1

    result_queue = Queue()

    # Start DB writer process
    db_writer = Process(target=save_results_worker, args=(result_queue,))
    db_writer.start()

    with Pool(processes=NUM_WORKERS) as pool:
        for result in pool.imap_unordered(create_image_worker, range(start_index, end_index)):
            result_queue.put(result)

    # Stop the DB writer
    result_queue.put('STOP')
    db_writer.join()