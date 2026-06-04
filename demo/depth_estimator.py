# depth_estimator.py
# Standalone monocular depth estimator using MiDaS for Mac demonstration

import cv2
import numpy as np
import torch

class DepthEstimator:
    def __init__(self):
        print("📦 [Depth Estimator] Loading MiDaS depth model from PyTorch Hub...")
        
        # 1. Determine device: MPS for Apple Silicon, CPU as fallback
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("⚡ [Depth Estimator] Device configured: Apple Silicon GPU (MPS)")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("⚡ [Depth Estimator] Device configured: NVIDIA GPU (CUDA)")
        else:
            self.device = torch.device("cpu")
            print("⚡ [Depth Estimator] Device configured: CPU")

        # 2. Load model
        # We use MiDaS_small (MiDaS v2.1) which is lightweight and fast for real-time inference on Mac
        try:
            self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            self.midas.to(self.device)
            self.midas.eval()
            
            # 3. Load transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.transform = midas_transforms.small_transform
            print("✅ [Depth Estimator] MiDaS model loaded successfully!")
        except Exception as e:
            print(f"❌ [Depth Estimator] Failed to load MiDaS model: {e}")
            raise e

    def estimate(self, bgr_frame):
        """
        Estimates depth for a BGR frame and returns both:
        1. A calibrated depth map in meters (float numpy array, same shape as input).
        2. A beautifully colorized depth map image (BGR, same shape as input) for display.
        """
        h_orig, w_orig = bgr_frame.shape[:2]
        
        # Convert BGR (OpenCV default) to RGB for MiDaS
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        
        # Apply transformation
        input_batch = self.transform(rgb_frame).to(self.device)
        
        with torch.no_grad():
            prediction = self.midas(input_batch)
            
            # Resize depth map back to original size
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(h_orig, w_orig),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            
            depth_map = prediction.cpu().numpy()
            
        # --- CALIBRATION TO METERS ---
        # MiDaS outputs relative inverse depth (disparity). Large values = close, small values = far.
        # Normalize to 0.0 - 1.0
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        
        if depth_max - depth_min > 1e-5:
            normalized_depth = (depth_map - depth_min) / (depth_max - depth_min)
        else:
            normalized_depth = np.zeros_like(depth_map)
            
        # Map normalized inverse depth (0.0 to 1.0) to meters (0.5m to 5.0m).
        # We invert the scale because 1.0 (close) should correspond to 0.5 meters,
        # and 0.0 (far) should correspond to 5.0 meters.
        min_meters = 0.5
        max_meters = 5.0
        depth_in_meters = min_meters + (1.0 - normalized_depth) * (max_meters - min_meters)
        
        # --- COLORIZATION FOR VISUALS ---
        # For visualization, we map 0.0-1.0 to 0-255 uint8.
        # Since standard JET colormap has Red as high (usually mapped to near) and Blue as low (mapped to far),
        # we colorize normalized_depth (where 1.0 is near -> Red, 0.0 is far -> Blue)
        depth_vis = (normalized_depth * 255.0).astype(np.uint8)
        colorized_depth = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        
        return depth_in_meters, colorized_depth

if __name__ == "__main__":
    # Test stub
    print("Testing DepthEstimator setup...")
    estimator = DepthEstimator()
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    depth_m, color_vis = estimator.estimate(dummy_frame)
    print(f"Estimation complete. Depth map range: {depth_m.min():.2f}m to {depth_m.max():.2f}m")
