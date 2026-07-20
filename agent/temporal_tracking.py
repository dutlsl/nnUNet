"""
Stage 3: Temporal Tracking and Eye Movement Classification Module
Applies Kalman Filtering and Savitzky-Golay smoothing on 200 Hz continuous pupil trajectories,
and classifies eye movements into Fixation, Saccade, and Blink.
"""
import numpy as np
from scipy.signal import savgol_filter

class KalmanGazeTracker:
    """2D Constant Velocity Kalman Filter for Eye Tracking."""
    def __init__(self, fps: float = 200.0, process_noise: float = 1e-2, measurement_noise: float = 1e-1):
        dt = 1.0 / fps
        self.dt = dt
        # State: [x, y, vx, vy]
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ], dtype=np.float64)
        
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float64)
        
        self.Q = np.eye(4) * process_noise
        self.R = np.eye(2) * measurement_noise
        
        self.x = np.zeros((4, 1), dtype=np.float64)
        self.P = np.eye(4) * 100.0
        self.initialized = False

    def update(self, z_x: float, z_y: float, is_blink: bool):
        if is_blink or np.isnan(z_x) or np.isnan(z_y):
            # Predict step only during blink/missing measurement
            self.x = self.F @ self.x
            self.P = self.F @ self.P @ self.F.T + self.Q
            return float(self.x[0, 0]), float(self.x[1, 0]), float(self.x[2, 0]), float(self.x[3, 0])
            
        z = np.array([[z_x], [z_y]], dtype=np.float64)
        
        if not self.initialized:
            self.x[0, 0] = z_x
            self.x[1, 0] = z_y
            self.initialized = True
            return z_x, z_y, 0.0, 0.0
            
        # Predict
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        
        # Update
        y = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        
        self.x = x_pred + K @ y
        self.P = (np.eye(4) - K @ self.H) @ P_pred
        
        return float(self.x[0, 0]), float(self.x[1, 0]), float(self.x[2, 0]), float(self.x[3, 0])

def process_sequence_trajectory(frames_data: list, fps: float = 200.0) -> list:
    """
    Takes a list of per-frame dicts from Stage 2 and performs Kalman filtering,
    Savitzky-Golay smoothing, velocity calculation, and movement classification.
    """
    tracker = KalmanGazeTracker(fps=fps)
    
    filtered_data = []
    raw_x = []
    raw_y = []
    
    # 1. Kalman Filter Pass
    for frame in frames_data:
        is_blink = frame.get("blink", False)
        cx = frame.get("pupil_cx", np.nan)
        cy = frame.get("pupil_cy", np.nan)
        
        kf_x, kf_y, kf_vx, kf_vy = tracker.update(cx, cy, is_blink)
        
        f_copy = dict(frame)
        f_copy["kf_cx"] = kf_x
        f_copy["kf_cy"] = kf_y
        f_copy["kf_vx"] = kf_vx
        f_copy["kf_vy"] = kf_vy
        
        filtered_data.append(f_copy)
        raw_x.append(kf_x)
        raw_y.append(kf_y)
        
    # 2. Savitzky-Golay Smoothing Pass (window length 11 frames = ~55 ms at 200Hz)
    n_frames = len(filtered_data)
    if n_frames >= 11:
        savgol_x = savgol_filter(raw_x, window_length=11, polyorder=2)
        savgol_y = savgol_filter(raw_y, window_length=11, polyorder=2)
    else:
        savgol_x = raw_x
        savgol_y = raw_y
        
    # 3. Calculate Velocity & Movement Classification
    saccade_threshold_px = 6.0  # px per frame at 200Hz (~300 px/sec)
    
    for i in range(n_frames):
        filtered_data[i]["smooth_cx"] = float(savgol_x[i])
        filtered_data[i]["smooth_cy"] = float(savgol_y[i])
        
        if i > 0:
            dx = savgol_x[i] - savgol_x[i-1]
            dy = savgol_y[i] - savgol_y[i-1]
            speed = np.sqrt(dx**2 + dy**2)
        else:
            speed = 0.0
            
        filtered_data[i]["speed_px_per_frame"] = float(speed)
        
        # Classification
        if filtered_data[i].get("blink", False):
            filtered_data[i]["event"] = "Blink"
        elif speed >= saccade_threshold_px:
            filtered_data[i]["event"] = "Saccade"
        else:
            filtered_data[i]["event"] = "Fixation"
            
    return filtered_data
