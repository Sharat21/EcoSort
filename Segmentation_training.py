import torch
import torchvision
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from sqlite3 import connect
from PIL import Image
import os
import torchvision.ops as ops
from scipy.optimize import linear_sum_assignment
# --------------------------
# Hyperparameters & Settings
# --------------------------
BATCH_SIZE = 4
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
IMAGE_SIZE = 2000
IMAGES_PER_BATCH = 10
LOAD_EXISTING_MODEL = True

# Extra loss weights for custom loss components.
base_alpha = 0.1       # Target weight for the custom mask loss (combined IoU/Dice + boundary loss)
lambda_boundary = 0.1  # Multiplier for the boundary loss term

# Threshold for using a matched pair in custom loss (if box IoU is too low, skip it)
iou_thresh_for_custom = 0.5

# --------------------------
# Custom Loss Functions for Masks
# --------------------------
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

# --------------------------
# Monitoring Function
# --------------------------
def monitor_predictions(predictions, targets):
    """
    Print basic statistics for each image in the batch.
    """
    for i, (pred, tgt) in enumerate(zip(predictions, targets)):
        num_pred = pred["boxes"].shape[0]
        num_gt = tgt["boxes"].shape[0]
        if num_pred > 0 and num_gt > 0:
            iou_matrix = ops.box_iou(pred["boxes"], tgt["boxes"])
            avg_iou = iou_matrix.mean().item()
        else:
            avg_iou = 0.0
        print(f"[Monitor] Image {i}: Pred Boxes = {num_pred}, GT Boxes = {num_gt}, Avg Box IoU = {avg_iou:.4f}")

# --------------------------
# Dataset Class
# --------------------------
class AugmentDataset(Dataset):
    """Custom dataset for loading images and annotations from an SQLite database."""
    def __init__(self, db_path='dataset.db', image_folder='./inputs/', batch_index=0):
        self.db_path = db_path
        self.image_folder = image_folder
        self.batch_index = batch_index
        self.data = self.load_data()
        if len(self.data) == 0:
            raise RuntimeError(f"No data loaded for batch {self.batch_index}. Check database.")
        print(f"Successfully loaded batch {self.batch_index}, {len(self.data)} images.")

    def load_data(self):
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
                continue
            full_size_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
            full_size_mask[y:y+h, x:x+w] = mask[:h, :w]
            bbox = self.get_tight_bounding_box(full_size_mask)
            if bbox is None:
                continue
            if image_index not in image_data:
                image_data[image_index] = {"objects": [], "file": f"image_{image_index}.jpg"}
            image_data[image_index]["objects"].append({"bbox": bbox, "mask": full_size_mask})
        return list(image_data.values())

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
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
        image = TF.to_tensor(image)
        masks, boxes = [], []
        for obj in image_info["objects"]:
            boxes.append(torch.tensor(obj["bbox"], dtype=torch.float32))
            mask = torch.tensor(obj["mask"], dtype=torch.float32)
            masks.append(mask.unsqueeze(0))
        if not masks:
            return image, {"boxes": torch.tensor([]), "masks": torch.tensor([])}
        return image, {
            "boxes": torch.stack(boxes),
            "masks": torch.cat(masks, dim=0).bool(),
            "labels": torch.ones((len(boxes),), dtype=torch.int64)
        }

# --------------------------
# Collate Function
# --------------------------
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return [], []
    images, targets = zip(*batch)
    return list(images), list(targets)

# --------------------------
# Training Functions with Scheduled Custom Loss and Monitoring
# --------------------------
def get_total_batches(db_path):
    conn = connect(db_path)
    c = conn.cursor()
    c.execute("SELECT COUNT(DISTINCT image_index) FROM images")
    total_images = c.fetchone()[0]
    conn.close()
    return (total_images // IMAGES_PER_BATCH) + (1 if total_images % IMAGES_PER_BATCH != 0 else 0)

def train_one_epoch(model, train_loader, optimizer, device, epoch, total_epochs, base_alpha=base_alpha, lambda_boundary=lambda_boundary, iou_thresh=iou_thresh_for_custom):
    """
    :param base_alpha: Target weight for the custom mask loss.
    :param epoch: Current epoch (starting from 1).
    :param total_epochs: Total number of epochs.
    :param lambda_boundary: Multiplier for the boundary loss.
    :param iou_thresh: Only include matched pairs with IoU >= this threshold.
    """
    model.train()
    epoch_loss = 0

    # Schedule custom loss weight: increase linearly over the first half of training.
    warmup_epochs = max(total_epochs // 2, 1)
    current_alpha = base_alpha * min(epoch / warmup_epochs, 1.0)
    print(f"[Epoch {epoch}] Current custom loss weight (alpha): {current_alpha:.4f}")

    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 1) Compute native Mask R-CNN loss.
        loss_dict = model(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())

        # 2) Evaluate predictions (no grad) to compute custom mask loss.
        model.eval()
        with torch.no_grad():
            predictions = model(images)
        model.train()

        # Monitoring: print bounding box stats.
        monitor_predictions(predictions, targets)

        custom_loss = 0.0

        # 3) For each image in the batch, match predicted boxes to GT boxes.
        for pred, tgt in zip(predictions, targets):
            if "masks" not in pred or len(pred["masks"]) == 0:
                print("[Monitor] No predicted masks for one image.")
                continue
            if "masks" not in tgt or len(tgt["masks"]) == 0:
                print("[Monitor] No GT masks for one image.")
                continue

            pred_boxes = pred["boxes"]
            gt_boxes = tgt["boxes"]

            # Compute IoU matrix between predicted boxes and GT boxes.
            iou_matrix = ops.box_iou(pred_boxes, gt_boxes).cpu().numpy()
            if iou_matrix.size == 0:
                continue
            row_indices, col_indices = linear_sum_assignment(-iou_matrix)
            avg_pair_iou = np.mean([iou_matrix[r, c] for r, c in zip(row_indices, col_indices)])
            print(f"[Monitor] Matched pairs: {len(row_indices)}; Average Box IoU: {avg_pair_iou:.4f}")

            # For each matched pair, only include if the matched box IoU >= threshold.
            for r, c in zip(row_indices, col_indices):
                if iou_matrix[r, c] < iou_thresh:
                    print(f"[Monitor] Skipping pair (r={r}, c={c}) with IoU {iou_matrix[r, c]:.4f} below threshold {iou_thresh}")
                    continue
                pred_mask = pred["masks"][r, 0]  # shape [H, W]
                gt_mask = tgt["masks"][c]        # shape [H, W]
                cm_loss = combined_mask_loss(pred_mask.unsqueeze(0), gt_mask.unsqueeze(0))
                b_loss = boundary_loss(pred_mask, gt_mask)
                pair_loss = cm_loss + lambda_boundary * b_loss
                custom_loss += pair_loss

        # Optionally average custom loss over the batch.
        # custom_loss /= len(images)

        total_loss += current_alpha * custom_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item()

    return epoch_loss / len(train_loader)

def train_model(model, device, num_epochs, db_path, base_alpha=base_alpha):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    total_batches = get_total_batches(db_path)
    for epoch in range(1, num_epochs+1):
        print(f"🔄 Starting epoch {epoch}/{num_epochs}")
        epoch_loss = 0
        for batch_index in range(total_batches):
            print(f"🔄 Loading batch {batch_index+1}/{total_batches}...")
            dataset = AugmentDataset(db_path=db_path, batch_index=batch_index)
            train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
            batch_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, num_epochs, base_alpha)
            epoch_loss += batch_loss
            print(f"Epoch {epoch}, Batch {batch_index+1}, Loss: {batch_loss:.4f}")
        avg_epoch_loss = epoch_loss / total_batches
        print(f"Epoch {epoch} completed. Average Loss: {avg_epoch_loss:.4f}")
        torch.save(model.state_dict(), f"segmentation_model_epoch{epoch}.pth")
    print("Training complete. Model saved.")

# --------------------------
# Main Execution
# --------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    if LOAD_EXISTING_MODEL:
        print("Loading custom trained model...")
        model = maskrcnn_resnet50_fpn(weights=None)
        num_classes = 2  # Background + Object
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
        model.load_state_dict(torch.load("segmentation_model.pth", map_location=device))
    else:
        print("Loading default Mask R-CNN model...")
        model = maskrcnn_resnet50_fpn(weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.to(device)
    train_model(model, device, NUM_EPOCHS, "dataset.db", base_alpha=base_alpha)
