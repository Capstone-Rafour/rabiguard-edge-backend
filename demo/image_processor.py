# image_processor.py
# Single image processing utility for Rafour Demo
# Integrates YOLO26n (Detection), YOLOE (Segmentation), and MiDaS (Depth)
# Saves individual model outputs and combined visualization to a separate output folder

import os
import sys
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Resolve paths
DEMO_DIR = Path(__file__).resolve().parent
ROOT_DIR = DEMO_DIR.parent
MODEL_DIR = ROOT_DIR / "models"

# Input/Output Directories
TEST_IMAGE_DIR = DEMO_DIR / "test_image"
TEST_OUTPUT_DIR = DEMO_DIR / "test_output"
INDOOR_TAGS_PATH = DEMO_DIR / "indoor_large_objects.txt"

# Create directories if they don't exist
TEST_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Add demo directory to path for depth_estimator
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from depth_estimator import DepthEstimator

class RafourImageProcessor:
    def __init__(self):
        print("📦 [Processor] Initializing Rafour Spatial AI Processor...")
        
        # 1. Load Depth Estimator
        try:
            self.depth_estimator = DepthEstimator()
        except Exception as e:
            print(f"❌ [Processor] Depth estimator initialization failed: {e}")
            sys.exit(1)

        # 2. Load YOLO Detection model (YOLO26n NCNN)
        yolo_model_path = MODEL_DIR / "yolo26n_ncnn_model"
        if not yolo_model_path.exists():
            print(f"⚠️ [Processor] YOLO NCNN model not found at {yolo_model_path}. Falling back to yolov8n.pt")
            self.yolo_model = YOLO("yolov8n.pt")
        else:
            print(f"✅ [Processor] Loading YOLO26n NCNN model from {yolo_model_path}")
            self.yolo_model = YOLO(str(yolo_model_path), task="detect")
            
        # 3. Load YOLOE Segmentation model (PT)
        yoloe_model_path = MODEL_DIR / "yoloe-26n-seg-pf.pt"
        if not yoloe_model_path.exists():
            print(f"⚠️ [Processor] YOLOE model not found at {yoloe_model_path}. Falling back to yolov8n-seg.pt")
            self.yoloe_model = YOLO("yolov8n-seg.pt")
        else:
            print(f"✅ [Processor] Loading YOLOE Segmentation model from {yoloe_model_path}")
            self.yoloe_model = YOLO(str(yoloe_model_path), task="segment")

        # 4. Load Indoor Tags for Filtering
        self.indoor_class_ids = []
        if INDOOR_TAGS_PATH.exists():
            with open(INDOOR_TAGS_PATH, "r", encoding="utf-8") as f:
                target_tags = set(line.strip().lower() for line in f if line.strip())
            
            # Map tags to model class IDs
            model_names = self.yoloe_model.names
            for class_id, name in model_names.items():
                if name.lower() in target_tags:
                    self.indoor_class_ids.append(class_id)
            
            print(f"✅ [Processor] Loaded {len(self.indoor_class_ids)} indoor/large object classes for filtering.")
        else:
            print(f"⚠️ [Processor] {INDOOR_TAGS_PATH} not found. Segmentation will show all classes.")

    def process(self, image_path):
        image_path = Path(image_path)
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"❌ [Processor] Could not read image at {image_path}")
            return
        
        h, w = img.shape[:2]
        base_name = image_path.stem
        print(f"🖼️ [Processor] Processing image: {image_path.name} ({w}x{h})")

        # --- 1. Depth Estimation ---
        print("🔍 [Processor] Running Depth Estimation...")
        depth_map_m, color_depth = self.depth_estimator.estimate(img)
        
        # Save depth output
        depth_out_path = TEST_OUTPUT_DIR / f"{base_name}_depth.jpg"
        cv2.imwrite(str(depth_out_path), color_depth)

        # --- 2. YOLO26n Detection ---
        print("🔍 [Processor] Running YOLO26n Detection (Person only)...")
        # Filter for 'person' class only (class index 0)
        det_results = self.yolo_model.predict(img, imgsz=640, verbose=False, classes=[0])
        det_overlay = det_results[0].plot() if det_results else img.copy()
        
        # Save detection output
        det_out_path = TEST_OUTPUT_DIR / f"{base_name}_yolo.jpg"
        cv2.imwrite(str(det_out_path), det_overlay)

        # --- 3. YOLOE Segmentation ---
        print("🔍 [Processor] Running YOLOE Segmentation (Filtered)...")
        # Filter for indoor/large object classes if IDs are available
        if self.indoor_class_ids:
            seg_results = self.yoloe_model.predict(img, imgsz=640, verbose=False, classes=self.indoor_class_ids)
        else:
            seg_results = self.yoloe_model.predict(img, imgsz=640, verbose=False)
            
        seg_overlay = seg_results[0].plot() if seg_results and seg_results[0].masks else img.copy()
        
        # Save segmentation output
        seg_out_path = TEST_OUTPUT_DIR / f"{base_name}_yoloe.jpg"
        cv2.imwrite(str(seg_out_path), seg_overlay)

        # --- 4. Create Combined Result ---
        target_w, target_h = 640, 480
        res_det = cv2.resize(det_overlay, (target_w, target_h))
        res_depth = cv2.resize(color_depth, (target_w, target_h))
        res_seg = cv2.resize(cv2.addWeighted(img, 0.5, seg_overlay, 0.5, 0), (target_w, target_h))
        
        depth_norm = cv2.normalize(depth_map_m, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        res_depth_raw = cv2.cvtColor(cv2.resize(depth_norm, (target_w, target_h)), cv2.COLOR_GRAY2BGR)

        top_row = np.hstack((res_det, res_depth))
        bottom_row = np.hstack((res_seg, res_depth_raw))
        combined = np.vstack((top_row, bottom_row))

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, "Detection (YOLO26n)", (20, 40), font, 1, (255, 255, 255), 2)
        cv2.putText(combined, "Depth (MiDaS)", (target_w + 20, 40), font, 1, (255, 255, 255), 2)
        cv2.putText(combined, "Segmentation (YOLOE)", (20, target_h + 40), font, 1, (255, 255, 255), 2)
        cv2.putText(combined, "Depth Map", (target_w + 20, target_h + 40), font, 1, (255, 255, 255), 2)

        combined_out_path = TEST_OUTPUT_DIR / f"{base_name}_result.jpg"
        cv2.imwrite(str(combined_out_path), combined)
        
        print(f"✅ [Processor] Outputs saved to {TEST_OUTPUT_DIR}")
        print(f"   - {det_out_path.name}")
        print(f"   - {seg_out_path.name}")
        print(f"   - {depth_out_path.name}")
        print(f"   - {combined_out_path.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rafour Spatial AI Image Processor")
    parser.add_argument("--input", "-i", type=str, help="Path to input image (optional, defaults to processing all in test_image/)")
    
    args = parser.parse_args()
    processor = RafourImageProcessor()

    if args.input:
        processor.process(args.input)
    else:
        # Process all common image formats in test_image directory
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
        image_files = []
        for ext in extensions:
            image_files.extend(TEST_IMAGE_DIR.glob(ext))
            
        if not image_files:
            print(f"ℹ️ No input images found in {TEST_IMAGE_DIR}. Please add some images there.")
        else:
            for img_path in image_files:
                processor.process(img_path)
