"""
Pre-extract LPW AVI videos into PNG frame directories for 100x faster dataset loading.
"""
import os
import glob
import cv2
from tqdm import tqdm

lpw_root = "/home/iulab1/PycharmProjects/transUnet/LPW"
extract_root = "/home/iulab1/PycharmProjects/transUnet/LPW_extracted"

avi_files = glob.glob(os.path.join(lpw_root, '*', '*.avi'))
print(f"Found {len(avi_files)} AVI video files in {lpw_root}")

os.makedirs(extract_root, exist_ok=True)

for avi_path in tqdm(avi_files):
    rel_path = os.path.relpath(avi_path, lpw_root)  # e.g., '1/1.avi'
    folder_id, file_name = os.path.split(rel_path)  # '1', '1.avi'
    file_id = os.path.splitext(file_name)[0]        # '1'

    out_dir = os.path.join(extract_root, folder_id, file_id)
    if os.path.exists(out_dir) and len(os.listdir(out_dir)) > 10:
        continue  # Already extracted

    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(avi_path)
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

print("Extraction complete!")
