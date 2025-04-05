
import torch
import torchvision
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from sqlite3 import connect
from PIL import Image
import os
import random
from torchvision.ops import nms

# **Hyperparameters**
MODEL_PATH = "segmentation_model.pth"
DB_PATH = "dataset.db"
IMAGE_FOLDER = "./inputs/"
SAVE_FOLDER = "./objects/"
CONFIDENCE_THRESHOLD = 0.60 # Save only bounding boxes with confidence > 70%
IMAGES_PER_BATCH = 20
BATCH_SIZE = 4
NUM_CLASSES = 2  # 1 object + background
IMAGE_SIZE = 2000  # Full image size to match training (from 480x640)
NMS_THRESHOLD = 0.4  # IOU threshold for NMS
# Ensure output directories exist
os.makedirs(SAVE_FOLDER, exist_ok=True)

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return [], []

    images, targets = zip(*batch)
    return list(images), list(targets)

# -------------------------------------
# **Dataset Class (Matching Training)**
# -------------------------------------
class EvaluationDataset(Dataset):
    """Loads images from the database for evaluation, structured identically to training."""
    def __init__(self, db_path=DB_PATH, image_folder=IMAGE_FOLDER, batch_index=0):
        self.db_path = db_path
        self.image_folder = image_folder
        self.batch_index = batch_index
        self.data = self.load_data()

        if len(self.data) == 0:
            raise RuntimeError(f"No data loaded for batch {self.batch_index}. Check database.")

        print(f"Loaded batch {self.batch_index}, {len(self.data)} images for evaluation.")

    def load_data(self):
        """Load image and annotation data from database exactly like training."""
        conn = connect(self.db_path)
        c = conn.cursor()

        offset = self.batch_index * IMAGES_PER_BATCH
        c.execute(
            "SELECT image_index, x_coord, y_coord, width, height, mask FROM images "
            "WHERE image_index BETWEEN ? AND ?",
            (offset, offset + IMAGES_PER_BATCH - 1)
        )
        data = c.fetchall()
        conn.close()

        image_data = {}
        for entry in data:
            image_index, x, y, w, h, mask_blob = entry
            mask = cv2.imdecode(np.frombuffer(mask_blob, np.uint8), cv2.IMREAD_GRAYSCALE)

            if mask is None:
                print(f"Warning: Skipping image {image_index} due to invalid mask")
                continue

            # **Create full-size mask (like training)**
            full_size_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
            full_size_mask[y:y+h, x:x+w] = mask[:h, :w]

            updated_bbox = self.get_tight_bounding_box(full_size_mask)
            if updated_bbox is None:
                print(f"Warning: No bounding box detected for image {image_index}")
                continue

            obj_x, obj_y, obj_w, obj_h = updated_bbox
            tight_bbox = [obj_x, obj_y, obj_x + obj_w, obj_y + obj_h]

            if image_index not in image_data:
                image_data[image_index] = {"objects": [], "file": f"image_{image_index}.jpg"}

            image_data[image_index]["objects"].append({
                "bbox": tight_bbox,
                "mask": full_size_mask
            })

        return list(image_data.values())

    def get_tight_bounding_box(self, mask):
        """Extract smallest bounding box that fits given segmentation mask."""
        coords = np.column_stack(np.where(mask > 0))
        if len(coords) == 0:
            return None
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        return (x_min, y_min, x_max - x_min, y_max - y_min)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Load an image and its corresponding masks and bounding boxes."""
        image_info = self.data[idx]
        image_path = os.path.join(self.image_folder, image_info["file"])

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

        image = F.to_tensor(image)
        masks, boxes = [], []

        for obj in image_info["objects"]:
            if obj["mask"] is None:
                continue

            boxes.append(obj["bbox"])
            mask = torch.tensor(obj["mask"], dtype=torch.float32)
            masks.append(mask.unsqueeze(0))

        if not masks:
            return image, {"boxes": torch.tensor([]), "masks": torch.tensor([]), "labels": torch.tensor([])}

        return image, {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "masks": torch.cat(masks, dim=0).bool(),
            "labels": torch.ones((len(boxes),), dtype=torch.int64)
        }

# -------------------------------------
# **Helper Function to Get Total Batches**
# -------------------------------------
def get_total_batches(db_path):
    """Get total number of batches based on images stored in database."""
    conn = connect(db_path)
    c = conn.cursor()
    c.execute("SELECT COUNT(DISTINCT image_index) FROM images")
    total_images = c.fetchone()[0]
    conn.close()
    return (total_images // IMAGES_PER_BATCH) + (1 if total_images % IMAGES_PER_BATCH != 0 else 0)

# -------------------------------------
# **Model Loading**
# -------------------------------------
def load_model(model_path, device):
    """Load trained Mask R-CNN model with compatible layers."""
    model = maskrcnn_resnet50_fpn(weights=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, NUM_CLASSES)

    in_channels = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_channels, 256, NUM_CLASSES)

    state_dict = torch.load(model_path, map_location=device)
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict() and model.state_dict()[k].shape == v.shape}
    model.load_state_dict(filtered_state_dict, strict=False)

    model.to(device)
    model.eval()
    print("Model loaded successfully with compatible layers.")
    return model

def filter_predictions(pred, min_threshold=0.3, max_threshold=CONFIDENCE_THRESHOLD, nms_threshold=NMS_THRESHOLD):
    """Filter predictions for a single image."""

    if "scores" not in pred or len(pred["scores"]) == 0:
        return {"boxes": torch.empty(0), "scores": torch.empty(0), "masks": torch.empty(0)}

    # Extract prediction values
    boxes = pred["boxes"]
    scores = pred["scores"]
    masks = pred["masks"] if "masks" in pred else None

    # **Step 1: Keep detections above a dynamic threshold**
    keep = scores >= max_threshold
    if keep.sum() == 0:  # If no detections, lower the threshold
        keep = scores >= min_threshold

    boxes = boxes[keep]
    scores = scores[keep]
    if masks is not None:
        masks = masks[keep]

    # **Step 2: Apply Non-Maximum Suppression (NMS)**
    if len(boxes) > 0:
        keep_nms = nms(boxes, scores, nms_threshold)
        boxes = boxes[keep_nms]
        scores = scores[keep_nms]
        if masks is not None:
            masks = masks[keep_nms]

    return {"boxes": boxes, "scores": scores, "masks": masks}

def visualize_predictions(image, predictions, save_path=None):
    """Overlay predicted masks and bounding boxes on the image."""

    image_np = image

    boxes = predictions["boxes"].cpu().numpy()
    scores = predictions["scores"].cpu().numpy()
    masks = predictions["masks"].cpu().numpy() if "masks" in predictions else []

    print(f"DEBUG: image_np shape: {image_np.shape}")  # Should be (H, W, 3)

    for i in range(len(scores)):
        if scores[i] < CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, boxes[i])
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # **Apply Segmentation Mask**
        if len(masks) > i:
            mask = masks[i, 0] > 0.3
            color = np.random.randint(100, 255, (1, 3), dtype=int).tolist()[0]
            image_np[mask] = (0.5 * image_np[mask] + 0.5 * np.array(color)).astype(np.uint8)

        # **Display Confidence Score**
        label = f"Object {i+1}: {scores[i]:.2f}"
        cv2.putText(image_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save output image
    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

    return image_np

# -------------------------------------
# **Updated `evaluate_model()`**
# -------------------------------------
def evaluate_model(model, device):
    total_batches = get_total_batches(DB_PATH)

    for batch_index in range(total_batches):
        dataset = EvaluationDataset(db_path=DB_PATH, batch_index=batch_index)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

        eval_idx = random.randint(0, len(dataset) - 1)  # Random image for IoU calculation
        print("eval :", eval_idx)
        for i, (images, targets) in enumerate(dataloader):
            images = [img.to(device) for img in images]

            with torch.no_grad():
                predictions = model(images)

            # Save one processed image per epoch with predictions
            if i == 0:
                for img_idx in range(len(images)):
                    output_filename = os.path.join(SAVE_FOLDER, f"eval_batch_{batch_index}_img_{img_idx}.jpg")
                    filtered_prediction = filter_predictions(predictions[img_idx])  # Remove extra list wrapping

                    # Convert PyTorch tensor (C, H, W) to OpenCV-compatible (H, W, C)
                    image_np = np.transpose(images[img_idx].cpu().numpy(), (1, 2, 0))
                    image_np = (image_np * 255).astype(np.uint8).copy()  # Convert to uint8 and make it contiguous
                    visualize_predictions(image_np, filtered_prediction, save_path=output_filename)

            # IoU Calculation for a random image in the batch
            if i == eval_idx:
                filtered_prediction = filter_predictions(predictions[0])  # Apply filtering to the first image in the batch

                pred_boxes = filtered_prediction["boxes"].cpu()
                gt_boxes = targets[0]["boxes"].cpu()

                if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                    iou = torchvision.ops.box_iou(pred_boxes, gt_boxes).mean().item()
                else:
                    iou = 0.0  # No detections
                print(f"Batch {batch_index} - Image {i}: IoU = {iou:.4f}")


    print("Evaluation complete.")

# -------------------------------------
# **Main Execution**
# -------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(MODEL_PATH, device)
    evaluate_model(model, device)