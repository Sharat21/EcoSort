import torch
import torchvision
import cv2
import numpy as np
from torchvision.transforms import functional as F
from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn
from PIL import Image, ImageDraw
import os

# **Set Parameters**
NUM_IMAGES_TO_VALIDATE = 5  # You can modify this number

# **Load Trained Model**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = maskrcnn_resnet50_fpn(weights=None)  # Initialize an empty model
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 2)

in_channels = model.roi_heads.mask_predictor.conv5_mask.in_channels
model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_channels, 256, 2)

model.load_state_dict(torch.load("segmentation_model.pth", map_location=device))
model.to(device)
model.eval()

# **Load Validation Images**
image_folder = "./inputs/"  # Change to your validation folder if needed
image_files = sorted(os.listdir(image_folder))[:NUM_IMAGES_TO_VALIDATE]  # Modify number of validation images

# **Inference on Images**
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    image = Image.open(image_path).convert("RGB")

    # Convert image to tensor
    image_tensor = F.to_tensor(image).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(image_tensor)[0]

    # Extract boxes, masks, and scores
    boxes = prediction["boxes"].cpu().numpy()
    masks = prediction["masks"].cpu().numpy()
    scores = prediction["scores"].cpu().numpy()

    # Draw results on the image
    draw = ImageDraw.Draw(image)

    for i in range(len(boxes)):
        if scores[i] > 0.5:  # Confidence threshold
            x1, y1, x2, y2 = boxes[i]
            draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=2)

            # Convert mask to binary and overlay it
            mask = masks[i, 0] > 0.5  # Thresholding the mask
            mask_image = Image.fromarray((mask * 255).astype(np.uint8))
            image.paste(mask_image, (0, 0), mask_image)

    # **Display or Save the Image**
    image.show()  # Shows the result (comment this if running in a remote environment)
    image.save(f"./outputs/{image_file}")  # Saves the output in an 'outputs' folder

print("✅ Inference complete. Processed images are saved in './outputs/'")
