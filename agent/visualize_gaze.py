"""
Gaze Trajectory Visualization Script
Plots Pupil Center (x, y) coordinates over time (200 Hz),
Speed profile (px/frame), and Fixation/Saccade/Blink event regions.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_gaze_trajectory(csv_path: str, save_path: str):
    df = pd.read_csv(csv_path)
    
    time_sec = df["frame_id"] * 0.005  # 200 Hz = 5 ms per frame
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    # 1. Pupil Center Position (X, Y)
    ax1.plot(time_sec, df["pupil_cx"], label="Pupil X (raw)", alpha=0.4, color="gray")
    ax1.plot(time_sec, df["smooth_cx"], label="Pupil X (Kalman+Savgol)", color="blue", linewidth=1.5)
    ax1.plot(time_sec, df["pupil_cy"], label="Pupil Y (raw)", alpha=0.4, color="lightcoral")
    ax1.plot(time_sec, df["smooth_cy"], label="Pupil Y (Kalman+Savgol)", color="red", linewidth=1.5)
    ax1.set_ylabel("Pixel Coord (px)")
    ax1.set_title("OpenEDS 200Hz Pupil Center Trajectory (Stage 2 & 3 Filtered)")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    
    # 2. Movement Speed (px / frame)
    ax2.plot(time_sec, df["speed_px_per_frame"], color="purple", linewidth=1.2)
    ax2.axhline(y=6.0, color="red", linestyle="--", label="Saccade Threshold (6 px/frame)")
    ax2.set_ylabel("Speed (px/frame)")
    ax2.set_title("Eye Movement Speed Profile")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    
    # 3. Classified Events Timeline
    colors = {"Fixation": "lightgreen", "Saccade": "orange", "Blink": "gray"}
    for idx, row in df.iterrows():
        t_start = row["frame_id"] * 0.005
        t_end = t_start + 0.005
        event = row["event"]
        c = colors.get(event, "white")
        ax3.axvspan(t_start, t_end, color=c, alpha=0.6)
        
    ax3.set_xlabel("Time (seconds)")
    ax3.set_ylabel("Event Classification")
    ax3.set_yticks([])
    
    # Custom legend for events
    handles = [plt.Rectangle((0,0),1,1, color=colors[e]) for e in ["Fixation", "Saccade", "Blink"]]
    ax3.legend(handles, ["Fixation (76.7%)", "Saccade (19.7%)", "Blink (3.6%)"], loc="upper right")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Gaze trajectory plot saved to: {save_path}")

if __name__ == "__main__":
    csv_file = "/home/iulab0/PycharmProjects/nnUNet/gaze_results/Validation_Part_1/Validation_Part_1_gaze_trajectory.csv"
    save_file = "/home/iulab0/PycharmProjects/nnUNet/gaze_results/Validation_Part_1/gaze_trajectory_plot.png"
    if Path(csv_file).exists():
        plot_gaze_trajectory(csv_file, save_file)
