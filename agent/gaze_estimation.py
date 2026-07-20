"""
Stage 2: Geometric Gaze Estimation Module (Pure SciPy/NumPy)
Extracts Pupil and Iris geometric parameters (centers, major/minor axes, orientation angle)
and estimates 2D gaze displacement vectors from segmentation masks without OpenCV dependency.
"""
import numpy as np
from scipy.ndimage import center_of_mass

def extract_gaze_features_from_mask(mask: np.ndarray) -> dict:
    """
    Given a 2D segmentation mask (0: BG, 1: Sclera, 2: Iris, 3: Pupil),
    returns a dictionary of geometric features and gaze parameters.
    """
    features = {
        "valid_frame": True,
        "blink": False,
        "pupil_cx": np.nan,
        "pupil_cy": np.nan,
        "pupil_area": 0,
        "iris_cx": np.nan,
        "iris_cy": np.nan,
        "iris_area": 0,
        "iris_major_axis": np.nan,
        "iris_minor_axis": np.nan,
        "iris_angle": np.nan,
        "gaze_x": np.nan,
        "gaze_y": np.nan,
    }
    
    pupil_mask = (mask == 3)
    iris_mask = (mask == 2)
    
    features["pupil_area"] = int(pupil_mask.sum())
    features["iris_area"] = int(iris_mask.sum())
    
    # Check for blink / missing pupil
    if features["pupil_area"] < 10:
        features["blink"] = True
        return features
        
    # Pupil Center of Mass
    cy, cx = center_of_mass(pupil_mask)
    features["pupil_cx"] = float(cx)
    features["pupil_cy"] = float(cy)
    
    # Iris Center and Shape Moments
    if features["iris_area"] >= 20:
        icy, icx = center_of_mass(iris_mask)
        features["iris_cx"] = float(icx)
        features["iris_cy"] = float(icy)
        
        # Calculate second-order central moments for Iris shape
        y_indices, x_indices = np.where(iris_mask)
        dx = x_indices - icx
        dy = y_indices - icy
        
        n_pts = len(x_indices)
        mu20 = np.sum(dx**2) / n_pts
        mu02 = np.sum(dy**2) / n_pts
        mu11 = np.sum(dx * dy) / n_pts
        
        # Eigenvalues of inertia tensor
        common = np.sqrt(((mu20 - mu02) / 2.0)**2 + mu11**2)
        lambda1 = (mu20 + mu02) / 2.0 + common
        lambda2 = max(0.0, (mu20 + mu02) / 2.0 - common)
        
        # Major and Minor axes (2 * std dev)
        features["iris_major_axis"] = float(4.0 * np.sqrt(lambda1))
        features["iris_minor_axis"] = float(4.0 * np.sqrt(lambda2))
        features["iris_angle"] = float(np.degrees(0.5 * np.arctan2(2.0 * mu11, mu20 - mu02)))
        
        # Gaze vector: relative offset between Pupil center and Iris center
        features["gaze_x"] = float(cx - icx)
        features["gaze_y"] = float(cy - icy)
    else:
        # If Iris is occluded, use image center (320, 200) as reference
        h, w = mask.shape
        features["gaze_x"] = float(cx - w / 2.0)
        features["gaze_y"] = float(cy - h / 2.0)
        
    return features
