import torch
import cv2
import numpy as np
from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight


# CONFIGURATION FILES (Update these paths)
CONFIG_PATH = "./config/nanodet-plus-m-1.5x_416.yml"  # Path to NanoDet YAML config
CHECKPOINT_PATH = "./model/nanodet-plus-m-1.5x_416.pth"  # Path to NanoDet model weights
IMAGE_PATH = "./ne.jpg"  # Path to input image

# 🟢 1. Load NanoDet Model
def load_nanodet_model(config_path, checkpoint_path):
    load_config(cfg, config_path)  # Load YAML config
    model = build_model(cfg.model)  # Build NanoDet model
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    local_rank = 0
    logger = Logger(local_rank, use_tensorboard=False)  # Set logging level
    logger.log('Press "Esc", "q" or "Q" to exit.')
    load_model_weight(model, checkpoint, logger)# Load model weights

      # Pass logger

    model.eval()  # Set model to evaluation mode
    return model

# 🟢 2. Preprocess Image
def preprocess_image(image_path, input_size=(416, 416)):
    image = cv2.imread(image_path)  # Load image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = cv2.resize(image, input_size)  # Resize for model
    image = image.astype(np.float32) / 255.0  # Normalize pixel values
    image = np.transpose(image, (2, 0, 1))  # Convert to (C, H, W)
    image = torch.tensor(image).unsqueeze(0)  # Add batch dimension
    return image

# 🟢 3. Run NanoDet Inference
def run_inference(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)  # Forward pass
    return output

# 🟢 4. Post-process & Draw Bounding Boxes
import cv2


def draw_boxes(image_path, detections):
    image = cv2.imread(image_path)  # Read image

    for det in detections:
        print(f"Detection Output: {det}")  # Debugging print

        # Ensure det is a Tensor and has expected shape
        if isinstance(det, torch.Tensor):
            det = det.cpu().numpy()  # Convert to NumPy for safety

        if len(det) >= 6:  # Ensure correct unpacking
            x1, y1, x2, y2, score, class_id = det[:6]

            # Convert potential multi-element tensors to Python scalars
            x1, y1, x2, y2 = map(lambda v: int(v) if isinstance(v, (int, float)) else int(v.item()), [x1, y1, x2, y2])

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"Class: {class_id} Score: {score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Detections", image)  # Show image
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# 🟢 5. MAIN FUNCTION
def main():
    model = load_nanodet_model(CONFIG_PATH, CHECKPOINT_PATH)  # Load model
    image_tensor = preprocess_image(IMAGE_PATH)  # Preprocess image
    detections = run_inference(model, image_tensor)  # Get detections
    draw_boxes(IMAGE_PATH, detections)  # Draw boxes

if __name__ == "__main__":
    main()
