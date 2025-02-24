import torch
import torchvision
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn, MaskRCNNPredictor
from torchvision.transforms import functional as F
from sqlite3 import connect
from PIL import Image
import os
import random

# **Hyperparameters**
BATCH_SIZE = 4
NUM_EPOCHS = 15  
LEARNING_RATE = 0.0005
NUM_CLASSES = 2  # 1 object + background
IMAGE_SIZE = 2000  # Full image size

# **IoU Calculation for Bounding Boxes**
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea)

# **IoU Calculation for Segmentation Masks**
def compute_mask_iou(pred_mask, gt_mask):
    gt_mask_resized = cv2.resize(gt_mask.astype(np.uint8), 
                                 (pred_mask.shape[1], pred_mask.shape[0]), 
                                 interpolation=cv2.INTER_NEAREST)

    pred_binary = pred_mask > 0.5
    gt_binary = gt_mask_resized > 0.5

    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    
    return intersection / union if union > 0 else 0

# **Set IoU Threshold**
IOU_THRESHOLD = 0.5  

# **Function to Compute Tight Bounding Box from Mask**
def get_tight_bounding_box(mask):
    coords = np.column_stack(np.where(mask > 0))
    if len(coords) == 0:
        return None  
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    return (x_min, y_min, x_max - x_min, y_max - y_min)

# **Dataset Class**
class AugmentDataset(Dataset):
    def __init__(self, db_path='dataset.db', image_folder='./inputs/'):
        self.db_path = db_path
        self.image_folder = image_folder
        self.data = self.load_data()

        if len(self.data) == 0:
            raise RuntimeError("❌ No data loaded! Check database path or contents.")

        print(f"✅ Successfully loaded {len(self.data)} images.")

    def load_data(self):
        """Load annotations from the SQLite database."""
        conn = connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT image_index, object_name, x_coord, y_coord, width, height, mask FROM images")
        data = c.fetchall()
        conn.close()

        image_data = {}
        for entry in data:
            image_index, object_name, x, y, w, h, mask_blob = entry
            mask = cv2.imdecode(np.frombuffer(mask_blob, np.uint8), cv2.IMREAD_GRAYSCALE)

            if mask is None:
                continue  

            full_size_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
            if y + h <= IMAGE_SIZE and x + w <= IMAGE_SIZE:
                full_size_mask[y:y+h, x:x+w] = mask
            else:
                print(f"⚠️ Warning: Object at ({x}, {y}, {w}, {h}) exceeds bounds.")

            updated_bbox = get_tight_bounding_box(full_size_mask)
            if updated_bbox is None:
                continue  

            obj_x, obj_y, obj_w, obj_h = updated_bbox
            tight_bbox = [obj_x, obj_y, obj_x + obj_w, obj_y + obj_h]

            if image_index not in image_data:
                image_data[image_index] = {"objects": [], "file": f"image_{image_index}.jpg"}

            image_data[image_index]["objects"].append({
                "class_name": object_name,
                "bbox": tight_bbox,
                "mask": full_size_mask
            })

        return list(image_data.values())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_info = self.data[idx]
        image_path = os.path.join(self.image_folder, image_info["file"])

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"❌ Error loading image {image_path}: {e}")
            return None  

        image = F.to_tensor(image)
        masks, boxes, labels = [], [], []

        for obj in image_info["objects"]:
            if obj["mask"] is None:
                continue  

            boxes.append(obj["bbox"])
            labels.append(1)  

            mask = torch.tensor(obj["mask"], dtype=torch.float32)
            masks.append(mask.unsqueeze(0))

        if not masks:
            return image, {"boxes": torch.tensor([]), "labels": torch.tensor([]), "masks": torch.tensor([])}

        return image, {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "masks": torch.cat(masks, dim=0).bool()
        }

# **Load dataset**
dataset = AugmentDataset()
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# **Dataset Preview**
sample_idx = random.randint(0, len(dataset) - 1)
sample_image, sample_target = dataset[sample_idx]
print(f"🔍 Sample Image Loaded: {dataset.data[sample_idx]['file']}")
print(f"📌 Number of Objects: {len(sample_target['boxes'])}")

# **Load Mask R-CNN Model**
model = maskrcnn_resnet50_fpn(weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, NUM_CLASSES)

in_channels = model.roi_heads.mask_predictor.conv5_mask.in_channels
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels, 256, NUM_CLASSES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# **Optimizer & Scheduler**
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# **Training Process**
model.train()
for epoch in range(NUM_EPOCHS):
    epoch_loss = 0
    for batch_idx, (images, targets) in enumerate(train_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item()

    lr_scheduler.step()
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss/len(train_loader):.4f}")
    torch.save(model.state_dict(), "segmentation_model.pth")

# Save Model
torch.save(model.state_dict(), "segmentation_model.pth")
print("✅ Training complete. Model saved as segmentation_model.pth")
