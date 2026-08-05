"""
Pre-extract LPW segment label MP4 videos into PNG mask frame directories for instant loading.
"""
import os
import glob
import cv2
from tqdm import tqdm

label_root = "/home/iulab1/PycharmProjects/nnUNet/Pupils in the wild improved"
out_label_root = "/home/iulab1/PycharmProjects/nnUNet/Pupils_in_the_wild_extracted"

mp4_files = glob.glob(os.path.join(label_root, '*.mp4'))
print(f"Found {len(mp4_files)} label MP4 files in {label_root}")

os.makedirs(out_label_root, exist_ok=True)

for mp4_path in tqdm(mp4_files):
    file_name = os.path.basename(mp4_path)  # e.g., 'folder-10_file-11_pupil.mp4'
    prefix = os.path.splitext(file_name)[0]  # 'folder-10_file-11_pupil'

    out_dir = os.path.join(out_label_root, prefix)
    if os.path.exists(out_dir) and len(os.listdir(out_dir)) > 10:
        continue

    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(mp4_path)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out_path = os.path.join(out_dir, f"{frame_idx:06d}.png")
        cv2.imwrite(out_path, gray)
        frame_idx += 1
    cap.release()

print("Label MP4 extraction complete!")
