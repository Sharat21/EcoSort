from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
from PIL import Image
import torch
from os import listdir, mkdir
from os.path import exists
import matplotlib.pyplot as plt
import matplotlib.patches as patches

SEGMENTATION_CONFIDENCE_THRESHOLD = 0.08
IDENTIFICATION_CONFIDENCE_THRESHOLD = 1000

model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
model.eval()

def identify_objects(image_index, x, y, w, h):
    from sqlite3 import connect

    conn = connect('dataset.db')
    c = conn.cursor()
    c.execute('SELECT * FROM images WHERE image_index = ?', (image_index,))
    objects = c.fetchall()
    conn.close()

    closest_match = None
    min_distance = float('inf')

    for obj in objects:
        _, obj_name, obj_x, obj_y, obj_w, obj_h = obj

        # Calculate the Euclidean distance between the given coordinates (x, y, w, h) and the object's coordinates (obj_x, obj_y, obj_w, obj_h)
        distance = ((x - obj_x) ** 2 + (y - obj_y) ** 2 + (w - obj_w) ** 2 + (h - obj_h) ** 2) ** 0.5

        if distance < min_distance:
            min_distance = distance
            closest_match = obj_name

    if abs(x - obj_x) <= IDENTIFICATION_CONFIDENCE_THRESHOLD and abs(y - obj_y) <= IDENTIFICATION_CONFIDENCE_THRESHOLD and \
       abs(w - obj_w) <= IDENTIFICATION_CONFIDENCE_THRESHOLD and abs(h - obj_h) <= IDENTIFICATION_CONFIDENCE_THRESHOLD:
        return closest_match
    else:
        return None

def segment_images_training():
    images = {}

    for file in listdir('./inputs'):
        file_name = file.split('.')[0]
        images[file_name] = {}
        images[file_name]['Image'] = Image.open("./inputs/" + file).convert("RGB")
        images[file_name]['Tensor'] = F.to_tensor(images[file_name]['Image']).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        for name, image in images.items():
            model.to(image['Tensor'])
            image['Prediction'] = model(image['Tensor'])[0]

    for name, image in images.items():
        boxes = image['Prediction']['boxes'].cpu().numpy()  # Bounding boxes
        scores = image['Prediction']['scores'].cpu().numpy()  # Confidence scores
        if not exists(f"./objects/{name}"): mkdir(f"./objects/{name}")
        fig, ax = plt.subplots(1)
        ax.imshow(image['Image'])
        file = open(f"./objects/{name}/objects.txt", 'w')

        for i in range(len(boxes)):
            if scores[i] > SEGMENTATION_CONFIDENCE_THRESHOLD: #Confidence threshold
                x1, y1, x2, y2 = boxes[i]
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                object_name = identify_objects(name.split('_')[1], x1, y1, x2 - x1, y2 - y1)

                if object_name is not None:
                    image['Image'].crop((x1, y1, x2, y2)).save(f"./objects/{name}/{object_name}.jpg")
                    file.write(f"Object {i}: ({x1}, {y1}), ({x2}, {y2} | Confidence Rating: {scores[i]} | Object Name: {object_name})\n")

        file.close()
        ax.axis('off')

        plt.savefig(f"./classifications/{name}", bbox_inches='tight', pad_inches=0)
        plt.close()

        print(f"Processed {name}")

def segment_images_prediction():
    images = {}

    for file in listdir('./inputs'):
        images[file] = {}
        images[file]['Image'] = Image.open("./inputs/" + file).convert("RGB")
        images[file]['Tensor'] = F.to_tensor(images[file]['Image']).unsqueeze(0)
        break

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        for name, image in images.items():
            model.to(image['Tensor'])
            image['Prediction'] = model(image['Tensor'])[0]

    for name, image in images.items():
        boxes = image['Prediction']['boxes'].cpu().numpy()  # Bounding boxes
        scores = image['Prediction']['scores'].cpu().numpy()  # Confidence scores
        if not exists(f"./objects/{name}"): mkdir(f"./objects/{name}")
        fig, ax = plt.subplots(1)
        ax.imshow(image['Image'])

        for i in range(len(boxes)):
            if scores[i] > 0.08: #Confidence threshold
                x1, y1, x2, y2 = boxes[i]
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                image['Image'].crop((x1, y1, x2, y2)).save(f"./objects/{name}/{i}.jpg")

        ax.axis('off')

        plt.savefig(f"./classifications/{name}", bbox_inches='tight', pad_inches=0)
        plt.close()
