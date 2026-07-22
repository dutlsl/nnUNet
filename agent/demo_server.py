"""
FastAPI Real-time Streaming Inference Server for TemporalUNet & Eye Tracking.

Features:
  - REST API: /predict (single frame), /reset (reset ConvLSTM memory)
  - WebSocket API: /ws/gaze_stream (ultra-low-latency real-time video stream)
  - Automatic preprocessing (Z-score normalization, padding)
  - Real-time pupil centroid (x, y) extraction & segmentation mask output
"""
import os
import sys
import io
import json
import base64
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(__file__))

from temporal_unet import TemporalUNet

app = FastAPI(
    title="TemporalUNet Real-Time Eye Tracking & Segmentation API",
    description="Serves TemporalUNet with ConvLSTM state memory over REST & WebSocket endpoints.",
    version="1.0.0"
)

# Global Model Container
MODEL: TemporalUNet = None
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MEAN_VAL = 86.45
STD_VAL = 39.94


@app.on_event("startup")
def load_model():
    global MODEL
    print("Initializing TemporalUNet for Real-Time Streaming Server...")
    model_dir = '/home/iulab0/PycharmProjects/nnUNet/nnUNet_results/Dataset600_OpenEDS2019/nnUNetTrainer_ImageNetPretrained__nnUNetPlans__2d'
    temporal_ckpt = '/home/iulab0/PycharmProjects/nnUNet/nnUNet_results/TemporalUNet_v1/checkpoint_best.pth'

    MODEL = TemporalUNet.from_pretrained(
        model_folder=model_dir,
        checkpoint_name='checkpoint_best.pth',
        num_classes=4,
        deep_supervision=False,
        device=torch.device(DEVICE),
    )
    
    ckpt = torch.load(temporal_ckpt, map_location=DEVICE, weights_only=False)
    if 'model_state_dict' in ckpt:
        MODEL.load_state_dict(ckpt['model_state_dict'])
    else:
        MODEL.load_state_dict(ckpt)
    
    MODEL.eval()
    MODEL.to(DEVICE)
    MODEL.reset_temporal_state()
    print("✅ TemporalUNet loaded and ready for inference!")


def process_frame(raw_img_np: np.ndarray):
    """
    Runs single-frame streaming inference on a 2D numpy array [H, W].
    Returns (pred_mask, pupil_center_y, pupil_center_x)
    """
    orig_h, orig_w = raw_img_np.shape[0], raw_img_np.shape[1]
    
    # 1. Preprocess: Normalization & Padding
    img_norm = (raw_img_np.astype(np.float32) - MEAN_VAL) / STD_VAL
    t_tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    target_h = ((orig_h + 63) // 64) * 64
    target_w = ((orig_w + 63) // 64) * 64
    pad_h = target_h - orig_h
    pad_w = target_w - orig_w
    if pad_h > 0 or pad_w > 0:
        t_tensor = torch.nn.functional.pad(t_tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)

    # 2. Forward Streaming
    with torch.no_grad():
        logits = MODEL.forward_streaming(t_tensor)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        pred_mask = logits.argmax(dim=1)[:, :orig_h, :orig_w].squeeze(0).cpu().numpy().astype(np.uint8)

    # 3. Calculate Pupil Center (Class 3)
    pupil_pts = np.argwhere(pred_mask == 3)
    if len(pupil_pts) > 0:
        cy, cx = pupil_pts.mean(axis=0)
        cy, cx = float(cy), float(cx)
    else:
        cy, cx = None, None

    return pred_mask, cy, cx


@app.get("/health")
def health_check():
    return {"status": "ok", "device": DEVICE, "model": "TemporalUNet_v1"}


@app.post("/reset")
def reset_state():
    """Reset ConvLSTM hidden state memory (call when starting a new user session)."""
    MODEL.reset_temporal_state()
    return {"status": "state_reset_success"}


@app.post("/predict_frame")
async def predict_frame(file: UploadFile = File(...)):
    """REST Endpoint: Accepts image file, returns segmentation & pupil coordinates."""
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('L')
    raw_np = np.array(image, dtype=np.uint8)

    pred_mask, cy, cx = process_frame(raw_np)

    return {
        "status": "success",
        "pupil_center": {"y": cy, "x": cx},
        "mask_shape": list(pred_mask.shape),
    }


@app.websocket("/ws/gaze_stream")
async def websocket_gaze_stream(websocket: WebSocket):
    """
    WebSocket Endpoint: Ultra-low-latency real-time video stream.
    Receives raw byte frames, returns JSON pupil coordinates & mask data.
    """
    await websocket.accept()
    MODEL.reset_temporal_state()
    print("WebSocket client connected. ConvLSTM memory reset.")

    try:
        while True:
            # Receive raw binary image bytes
            data = await websocket.receive_bytes()
            image = Image.open(io.BytesIO(data)).convert('L')
            raw_np = np.array(image, dtype=np.uint8)

            pred_mask, cy, cx = process_frame(raw_np)

            # Send back real-time pupil coordinates
            response_data = {
                "pupil_center": {"y": cy, "x": cx},
                "pupil_detected": (cy is not None),
            }
            await websocket.send_text(json.dumps(response_data))

    except WebSocketDisconnect:
        print("WebSocket client disconnected.")
    except Exception as e:
        print(f"WebSocket Error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
