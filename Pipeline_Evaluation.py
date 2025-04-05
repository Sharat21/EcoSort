import torch
import torchvision
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision.models.detection import maskrcnn_resnet50_fpn
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from sqlite3 import connect
from PIL import Image
import os
import torchvision.ops as ops
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from torchvision import transforms

import time

# ====================================================
# Classification Model Settings and Functions
# ====================================================

# Label mapping for classification
labels = {
    'cardboard': 0,
    'metal': 1,
    'plastic': 2,
    'trash': 3,
    'glass': 4,
    'paper': 5
}
label_list = list(labels.keys())

# Classification transform (resize, normalize, etc.)
cls_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def create_classification_model():
    """
    Creates an EfficientNet-B3 model with a modified classifier for our classes.
    """
    from torch import nn
    from torchvision.models import efficientnet_b3
    model = efficientnet_b3(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[1].in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, len(labels))
    )
    return model

# ====================================================
# Segmentation Functions and Dataset (Validation)
# ====================================================

# For segmentation we assume full-size images (square) of size IMAGE_SIZE
IMAGE_SIZE = 2000

def iou_loss(pred_mask, target_mask, eps=1e-6):
    intersection = (pred_mask * target_mask).sum(dim=(1, 2))
    union = (pred_mask + target_mask).sum(dim=(1, 2)) - intersection
    iou = (intersection + eps) / (union + eps)
    return (1 - iou).mean()

def dice_loss(pred_mask, target_mask, eps=1e-6):
    pred_mask = (pred_mask > 0.5).float()
    target_mask = target_mask.float()
    intersection = (pred_mask * target_mask).sum(dim=(1, 2))
    total_area = pred_mask.sum(dim=(1, 2)) + target_mask.sum(dim=(1, 2))
    dice = (2 * intersection + eps) / (total_area + eps)
    return 1 - dice.mean()

def combined_mask_loss(pred_mask, target_mask):
    return 0.5 * iou_loss(pred_mask, target_mask) + 0.5 * dice_loss(pred_mask, target_mask)

def boundary_loss(pred_mask, target_mask):
    """
    Compute an L1 loss on the edges of the predicted and ground-truth masks.
    Uses a Sobel operator.
    """
    sobel_kernel = torch.tensor([[-1, -2, -1],
                                 [ 0,  0,  0],
                                 [ 1,  2,  1]], dtype=torch.float32, device=pred_mask.device)
    sobel_kernel = sobel_kernel.unsqueeze(0).unsqueeze(0)  # [1,1,3,3]
    pred_mask = pred_mask.unsqueeze(0).unsqueeze(0).float()  # [1,1,H,W]
    target_mask = target_mask.unsqueeze(0).unsqueeze(0).float()  # [1,1,H,W]
    pred_edges = F.conv2d(pred_mask, sobel_kernel, padding=1)
    target_edges = F.conv2d(target_mask, sobel_kernel, padding=1)
    return F.l1_loss(pred_edges, target_edges)

def mask_iou(pred_mask, target_mask, eps=1e-6):
    """Compute IoU for binary masks."""
    pred_mask_bin = (pred_mask > 0.5).float()
    target_mask = target_mask.float()
    intersection = (pred_mask_bin * target_mask).sum().item()
    union = pred_mask_bin.sum().item() + target_mask.sum().item() - intersection
    return (intersection + eps) / (union + eps)

class ValidationDataset(Dataset):
    """
    Loads all images (each with one or more objects) from the SQLite database.
    Each sample is one image along with its annotations.
    """
    def __init__(self, db_path='dataset.db', image_folder='./inputs/'):
        self.db_path = db_path
        self.image_folder = image_folder
        self.data = self.load_data()
        if len(self.data) == 0:
            raise RuntimeError("No data loaded from database.")
        print(f"Successfully loaded {len(self.data)} images.")

    def load_data(self):
        conn = connect(self.db_path)
        c = conn.cursor()
        # Get all distinct image indices
        c.execute("SELECT DISTINCT image_index FROM images ORDER BY image_index ASC LIMIT 15")
        image_indices = [row[0] for row in c.fetchall()]
        data = []
        for image_index in image_indices:
            # Retrieve object_id, object_name, and other fields for each object
            c.execute(
                "SELECT object_id, object_name, x_coord, y_coord, width, height, mask FROM images WHERE image_index = ?",
                (image_index,)
            )
            entries = c.fetchall()
            image_data = {"objects": [], "file": f"image_{image_index}.jpg"}
            for entry in entries:
                object_id, object_name, x, y, w, h, mask_blob = entry
                # Decode the compressed mask from the blob using OpenCV.
                mask = cv2.imdecode(np.frombuffer(mask_blob, np.uint8), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue
                # Create a full–size mask for the image.
                full_size_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
                full_size_mask[y:y+h, x:x+w] = mask[:h, :w]
                bbox = self.get_tight_bounding_box(full_size_mask)
                if bbox is None:
                    continue
                image_data["objects"].append({
                    "bbox": bbox,
                    "mask": full_size_mask,
                    "object_id": object_id,
                    "object_name": object_name
                })
            if image_data["objects"]:
                data.append(image_data)
        conn.close()
        return data

    def get_tight_bounding_box(self, mask):
        coords = np.column_stack(np.where(mask > 0))
        if len(coords) == 0:
            return None
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        return [x_min, y_min, x_max, y_max]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_info = self.data[idx]
        image_path = os.path.join(self.image_folder, image_info["file"])
        try:
            pil_image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
        image_tensor = TF.to_tensor(pil_image)
        boxes = []
        masks = []
        object_ids = []
        object_names = []
        for obj in image_info["objects"]:
            boxes.append(torch.tensor(obj["bbox"], dtype=torch.float32))
            mask = torch.tensor(obj["mask"], dtype=torch.float32)
            masks.append(mask.unsqueeze(0))
            object_ids.append(obj["object_id"])
            object_names.append(obj["object_name"])
        if not boxes:
            target = {
                "boxes": torch.tensor([]), 
                "masks": torch.tensor([]), 
                "labels": torch.tensor([]),
                "object_ids": torch.tensor([]),
                "object_names": [],
                "file_path": image_path
            }
        else:
            target = {
                "boxes": torch.stack(boxes),
                "masks": torch.cat(masks, dim=0).bool(),
                "labels": torch.ones((len(boxes),), dtype=torch.int64),
                "object_ids": torch.tensor(object_ids, dtype=torch.int64),
                "object_names": object_names,
                "file_path": image_path
            }
        return image_tensor, target

# --------------------------
# Monitoring Function for Segmentation Metrics
# --------------------------
def monitor_inference(prediction, target, iou_thresh=0.5):
    """
    For a single image, prints statistics and returns segmentation metrics.
    Computes both average box IoU and average mask IoU using linear assignment.
    """
    pred_boxes = prediction.get("boxes", torch.empty(0))
    gt_boxes = target.get("boxes", torch.empty(0))
    num_pred = pred_boxes.shape[0]
    num_gt = gt_boxes.shape[0]
    avg_box_iou = 0.0
    avg_mask_iou = 0.0
    if num_pred > 0 and num_gt > 0:
        iou_matrix = ops.box_iou(pred_boxes, gt_boxes)
        iou_np = iou_matrix.cpu().numpy()
        row_indices, col_indices = linear_sum_assignment(-iou_np)
        matched_box_ious = [iou_np[r, c] for r, c in zip(row_indices, col_indices)]
        avg_box_iou = np.mean(matched_box_ious) if matched_box_ious else 0.0

        mask_ious = []
        for r, c in zip(row_indices, col_indices):
            if iou_np[r, c] < iou_thresh:
                print(f"[Monitor] Skipping mask comparison for pair (r={r}, c={c}) with box IoU {iou_np[r, c]:.4f}")
                continue
            pred_mask = prediction["masks"][r, 0]  # shape [H, W]
            gt_mask = target["masks"][c]           # shape [H, W]
            miou = mask_iou(pred_mask, gt_mask)
            mask_ious.append(miou)
        avg_mask_iou = np.mean(mask_ious) if mask_ious else 0.0
    print(f"[Monitor] Predicted Boxes: {num_pred}, GT Boxes: {num_gt}, Avg Box IoU: {avg_box_iou:.4f}, Avg Mask IoU: {avg_mask_iou:.4f}")
    return num_pred, num_gt, avg_box_iou, avg_mask_iou

# --------------------------
# Classification Evaluation for Each Detected Object
# --------------------------
def evaluate_classification_for_image(pil_image, prediction, target, classification_model, device, iou_thresh=0.5):
    """
    For a single image, for each segmentation-predicted bounding box:
      - Find the best matching ground-truth (via box IoU).
      - If the max IoU is above threshold, crop the region from the original image,
        run classification, and compare against the ground-truth label.
    Returns a list of classification results.
    """
    results = []
    pred_boxes = prediction.get("boxes", torch.empty(0))
    gt_boxes = target.get("boxes", torch.empty(0))
    if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
        return results
    
    for j in range(pred_boxes.shape[0]):
        # Compute IoU between the predicted box and all ground truth boxes.
        box_j = pred_boxes[j].unsqueeze(0)  # shape [1,4]
        iou_vec = ops.box_iou(box_j, gt_boxes)[0]  # shape [num_gt]
        max_iou, gt_idx = torch.max(iou_vec, dim=0)
        if max_iou.item() < iou_thresh:
            # Skip if not a good match.
            continue
        # Crop using the predicted box.
        box_np = pred_boxes[j].cpu().numpy()  # [x_min, y_min, x_max, y_max]
        cropped_region = pil_image.crop((box_np[0], box_np[1], box_np[2], box_np[3]))
        # Prepare the cropped image for classification.
        cropped_tensor = cls_transform(cropped_region).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = classification_model(cropped_tensor)
        pred_class_idx = outputs.argmax(dim=1).item()
        predicted_label = label_list[pred_class_idx]
        confidence = torch.softmax(outputs, dim=1)[0, pred_class_idx].item()
        ground_truth = target["object_names"][gt_idx.item()]
        correct = (predicted_label.lower() == ground_truth.lower())
        results.append({
            "predicted": predicted_label,
            "confidence": confidence,
            "ground_truth": ground_truth,
            "correct": correct,
            "pred_index": j,
            "gt_index": gt_idx.item(),
            "iou": max_iou.item()
        })
    return results

# --------------------------
# Validation Function (Combined Segmentation & Classification)
# --------------------------
def validate_model(seg_model, cls_model, device, db_path='dataset.db', image_folder='./inputs/', iou_thresh=0.5, score_thresh=0.6):
    """
    For each image in the database:
      - Runs segmentation (only keeping predictions with a score above score_thresh).
      - Computes segmentation metrics.
      - For each segmentation detection, finds the closest ground truth based on IoU.
      - If a good match is found, the predicted box is used to crop the region from the image,
        which is then saved to the "output" folder and passed to the classification model.
      - Aggregates all classification and segmentation metrics across images.
    """
    seg_model.eval()
    if cls_model is not None:
        cls_model.eval()
    dataset = ValidationDataset(db_path=db_path, image_folder=image_folder)
    all_classification_results = []
    all_seg_box_ious = []
    all_seg_mask_ious = []
    total_images = 0

    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)

    for i in range(len(dataset)):
        sample = dataset[i]
        if sample is None:
            continue
        total_images += 1
        image_tensor, target = sample
        # Ensure the target contains the file path (ValidationDataset should add "file_path")
        pil_image = Image.open(target["file_path"]).convert("RGB")
        image_tensor = image_tensor.to(device)
        with torch.no_grad():
            # Segmentation model expects a list of images.
            prediction = seg_model([image_tensor])[0]
        
        # --- Filter predictions by segmentation score ---
        if "scores" in prediction and prediction["scores"].numel() > 0:
            valid_indices = [j for j, s in enumerate(prediction["scores"]) if s.item() > score_thresh]
            if len(valid_indices) == 0:
                print(f"[Image {i}] No predictions with score > {score_thresh}")
                continue
            prediction["boxes"] = prediction["boxes"][valid_indices]
            prediction["masks"] = prediction["masks"][valid_indices]
            prediction["scores"] = prediction["scores"][valid_indices]
        
        # --- Compute segmentation matching metrics ---
        num_pred, num_gt, avg_box_iou, avg_mask_iou = monitor_inference(prediction, target, iou_thresh=iou_thresh)
        print(f"[Image {i}] Segmentation: {num_pred} predicted, {num_gt} GT, Avg Box IoU: {avg_box_iou:.4f}, Avg Mask IoU: {avg_mask_iou:.4f}")
        all_seg_box_ious.append(avg_box_iou)
        all_seg_mask_ious.append(avg_mask_iou)

        # --- Evaluate classification for each detected object ---
        cls_results = evaluate_classification_for_image(pil_image, prediction, target, cls_model, device, iou_thresh=iou_thresh)
        # Save each cropped region and print classification details.
        for res in cls_results:
            j = res["pred_index"]
            box = prediction["boxes"][j].cpu().numpy()
            cropped_region = pil_image.crop((box[0], box[1], box[2], box[3]))
            cropped_path = os.path.join(output_folder, f"image_{i}_object_{j}.png")
            cropped_region.save(cropped_path)
            print(f"[Output] Saved cropped region to {cropped_path}")
            correctness = "correct" if res["correct"] else "incorrect"
            print(f"[Classification] Object '{res['ground_truth']}': Predicted='{res['predicted']}' ({correctness}), Confidence={res['confidence']:.4f}, IoU={res['iou']:.4f}")
        all_classification_results.extend(cls_results)
    
    # --- Aggregate and Print Final Metrics ---
    final_avg_box_iou = np.mean(all_seg_box_ious) if all_seg_box_ious else 0.0
    final_avg_mask_iou = np.mean(all_seg_mask_ious) if all_seg_mask_ious else 0.0

    if all_classification_results:
        corrects = [1 if r["correct"] else 0 for r in all_classification_results]
        overall_accuracy = sum(corrects) / len(corrects)
        correct_confidences = [r["confidence"] for r in all_classification_results if r["correct"]]
        incorrect_confidences = [r["confidence"] for r in all_classification_results if not r["correct"]]
        highest_confidence = max([r["confidence"] for r in all_classification_results])
        lowest_confidence = min([r["confidence"] for r in all_classification_results])
        avg_correct_conf = np.mean(correct_confidences) if correct_confidences else 0.0
        avg_incorrect_conf = np.mean(incorrect_confidences) if incorrect_confidences else 0.0
    else:
        overall_accuracy = 0.0
        highest_confidence = 0.0
        lowest_confidence = 0.0
        avg_correct_conf = 0.0
        avg_incorrect_conf = 0.0

    print("\n===== Final Model Metrics =====")
    print(f"Total Images Processed: {total_images}")
    print(f"Segmentation - Average Box IoU: {final_avg_box_iou:.4f}, Average Mask IoU: {final_avg_mask_iou:.4f}")
    print(f"Classification - Overall Accuracy: {overall_accuracy*100:.2f}%")
    print(f"Highest Classification Confidence: {highest_confidence:.4f}, Lowest Classification Confidence: {lowest_confidence:.4f}")
    print(f"Average Confidence (Correct Predictions): {avg_correct_conf:.4f}")
    print(f"Average Confidence (Incorrect Predictions): {avg_incorrect_conf:.4f}")

# ====================================================
# Main Execution
# ====================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    # --------------------------
    # Load Segmentation Model
    # --------------------------
    print("Loading trained segmentation model for validation...")
    seg_model = maskrcnn_resnet50_fpn(weights=None)
    num_classes = 2  # Background + Object
    in_features = seg_model.roi_heads.box_predictor.cls_score.in_features
    seg_model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    seg_model.load_state_dict(torch.load("segmentation_model_epoch6.pth", map_location=device))
    seg_model.to(device)

    # --------------------------
    # Load Classification Model
    # --------------------------
    print("Loading trained classification model for evaluation...")
    cls_model = create_classification_model()
    cls_model.load_state_dict(torch.load("best_model_epoch_3.pth", map_location=device))
    cls_model.to(device)

    # --------------------------
    # Run Validation (Combined) with Timing
    # --------------------------
    start_time = time.time()  # Start timing
    validate_model(seg_model, cls_model, device, db_path="dataset.db", image_folder="./inputs/", iou_thresh=0.5)
    elapsed_time = time.time() - start_time  # End timing
    print(f"Total time elapsed: {elapsed_time:.2f} seconds")