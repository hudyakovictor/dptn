import sys
import os
import torch
import numpy as np
import cv2
import logging
from pathlib import Path
from torchvision import transforms

# Path to the external repository
_hpe_env = os.environ.get("DUTIN_HPE_PATH", "").strip()
_possible_hpe_paths = [
    Path(_hpe_env) if _hpe_env else None,
    Path(__file__).resolve().parents[2] / "core" / "head-pose-estimation",
    Path(__file__).resolve().parents[1] / "core" / "head-pose-estimation",
    Path("./backend/core/head-pose-estimation"),
    Path("/Users/victorkhudyakov/dutin/core/head-pose-estimation"),
]
EXTERNAL_REPO_PATH = str(
    next((p for p in _possible_hpe_paths if p is not None and p.exists()), _possible_hpe_paths[-1])
)

# Ensure the repository is in sys.path so we can import from it
if EXTERNAL_REPO_PATH not in sys.path:
    sys.path.insert(0, EXTERNAL_REPO_PATH)

try:
    from models import get_model, SCRFD
    from utils.general import compute_euler_angles_from_rotation_matrices
except ImportError as e:
    logging.warning(f"Could not import head-pose-estimation library: {e}")
    SCRFD = None
    get_model = None

def expand_bbox(x_min, y_min, x_max, y_max, factor=0.2):
    """Expand the bounding box by a given factor."""
    width = x_max - x_min
    height = y_max - y_min

    x_min_new = x_min - int(factor * height)
    y_min_new = y_min - int(factor * width)
    x_max_new = x_max + int(factor * height)
    y_max_new = y_max + int(factor * width)

    return max(0, x_min_new), max(0, y_min_new), x_max_new, y_max_new

def pre_process(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image)
    image_batch = image.unsqueeze(0)
    return image_batch

class HighResHeadPoseEstimator:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
        
    def _initialize(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.face_detector = None
        self.head_pose = None
        
        if SCRFD is None or get_model is None:
            logging.error("head-pose-estimation models not available due to import error.")
            return

        try:
            det_path = os.path.join(EXTERNAL_REPO_PATH, "weights", "det_10g.onnx")
            self.face_detector = SCRFD(model_path=det_path)
            logging.info("HighResHeadPoseEstimator: Face Detection model weights loaded.")
        except Exception as e:
            logging.error(f"HighResHeadPoseEstimator: Failed to load SCRFD face detector: {e}")

        try:
            weights_path = os.path.join(EXTERNAL_REPO_PATH, "weights", "mobilenetv3_large.pt")
            self.head_pose = get_model("mobilenetv3_large", num_classes=6, pretrained=False)
            state_dict = torch.load(weights_path, map_location=self.device)
            self.head_pose.load_state_dict(state_dict)
            self.head_pose.to(self.device)
            self.head_pose.eval()
            logging.info("HighResHeadPoseEstimator: Head Pose Estimation model loaded.")
        except Exception as e:
            logging.error(f"HighResHeadPoseEstimator: Failed to load head pose model: {e}")

    def predict(self, image_path_or_array):
        """
        Returns a dictionary with 'yaw', 'pitch', 'roll' in degrees or None if no face detected.
        """
        if self.face_detector is None or self.head_pose is None:
            return None

        if isinstance(image_path_or_array, (str, Path)):
            frame = cv2.imread(str(image_path_or_array))
        else:
            frame = image_path_or_array
            
        if frame is None:
            return None

        with torch.no_grad():
            bboxes, keypoints = self.face_detector.detect(frame)
            if len(bboxes) == 0:
                return None
                
            # Pick the largest face if multiple
            if len(bboxes) > 1:
                areas = [(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in bboxes]
                best_idx = np.argmax(areas)
                bbox = bboxes[best_idx]
            else:
                bbox = bboxes[0]

            x_min, y_min, x_max, y_max = map(int, bbox[:4])
            x_min, y_min, x_max, y_max = expand_bbox(x_min, y_min, x_max, y_max)

            # Ensure bounds are within image
            h, w = frame.shape[:2]
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)
            
            if x_max <= x_min or y_max <= y_min:
                return None

            image_crop = frame[y_min:y_max, x_min:x_max]
            image_tensor = pre_process(image_crop).to(self.device)

            rotation_matrix = self.head_pose(image_tensor)
            euler = np.degrees(compute_euler_angles_from_rotation_matrices(rotation_matrix))
            
            # euler format: [pitch, yaw, roll]
            p_pred_deg = float(euler[:, 0].cpu()[0])
            y_pred_deg = float(euler[:, 1].cpu()[0])
            r_pred_deg = float(euler[:, 2].cpu()[0])

            return {
                "yaw": y_pred_deg,
                "pitch": p_pred_deg,
                "roll": r_pred_deg
            }
