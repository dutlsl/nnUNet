"""
FastAPI Demo Server for OpenEDS Segmentation & Gaze Tracking
Provides REST API endpoints for real-time eye segmentation and gaze estimation.
"""
import os
import io
import base64
import torch
import numpy as np
from PIL import Image
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from acvl_utils.cropping_and_padding.padding import pad_nd_image

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from gaze_estimation import extract_gaze_features_from_mask
from temporal_tracking import process_sequence_trajectory

app = FastAPI(
    title="OpenEDS Gaze Tracking Demo API",
    description="Real-time 4-class Eye Segmentation & Temporal Gaze Tracking API",
    version="1.0.0"
)

# Global model predictor
PREDICTOR = None
NETWORK = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_DIR = os.environ.get(
    'MODEL_DIR',
    '/app/nnUNet_results/Dataset600_OpenEDS2019/nnUNetTrainer_ImageNetPretrained__nnUNetPlans__2d'
)

@app.on_event("startup")
def load_model():
    global PREDICTOR, NETWORK, DEVICE
    print(f"Loading OpenEDS nnUNet Model on {DEVICE}...")
    if os.path.exists(MODEL_DIR):
        try:
            PREDICTOR = nnUNetPredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=False,
                perform_everything_on_device=True,
                device=DEVICE,
                verbose=False
            )
            PREDICTOR.initialize_from_trained_model_folder(
                MODEL_DIR,
                use_folds=(0,),
                checkpoint_name='checkpoint_best.pth'
            )
            NETWORK = PREDICTOR.network.to(DEVICE)
            NETWORK.eval()
            print("Model successfully loaded and ready for inference!")
        except Exception as e:
            print(f"Warning: Could not initialize model from {MODEL_DIR}: {e}")
    else:
        print(f"Warning: MODEL_DIR {MODEL_DIR} not found. Running in mock/dry-run mode.")

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available(),
        "model_loaded": NETWORK is not None
    }

def _predict_mask_from_pil(img: Image.Image) -> np.ndarray:
    """Helper to run model forward pass on a PIL Grayscale image."""
    img_gray = img.convert("L")
    img_np = np.array(img_gray, dtype=np.float32)
    
    mean_val = img_np.mean()
    std_val = img_np.std()
    img_norm = (img_np - mean_val) / std_val if std_val > 1e-8 else img_np - mean_val
    
    img_padded, slicer = pad_nd_image(img_norm[None, None], new_shape=(448, 640), return_slicer=True, mode='constant')
    img_tensor = torch.from_numpy(img_padded).to(DEVICE)
    
    with torch.no_grad():
        logits = NETWORK(img_tensor)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        mask = logits.argmax(dim=1)[0, slicer[2], slicer[3]].cpu().numpy().astype(np.uint8)
        
    return mask

@app.post("/predict_gaze")
async def predict_gaze(file: UploadFile = File(...)):
    """Accepts an eye image file and returns 2D pupil/iris gaze parameters."""
    if NETWORK is None:
        raise HTTPException(status_code=503, detail="Model not initialized.")
        
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        mask = _predict_mask_from_pil(img)
        features = extract_gaze_features_from_mask(mask)
        return JSONResponse(content=features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_sequence_gaze")
async def predict_sequence_gaze(files: List[UploadFile] = File(...)):
    """Accepts multiple sequential frame images (200Hz) and returns temporal gaze trajectory."""
    if NETWORK is None:
        raise HTTPException(status_code=503, detail="Model not initialized.")
        
    try:
        raw_results = []
        for idx, file in enumerate(files):
            contents = await file.read()
            img = Image.open(io.BytesIO(contents))
            mask = _predict_mask_from_pil(img)
            feat = extract_gaze_features_from_mask(mask)
            feat["frame_id"] = idx
            feat["filename"] = file.filename
            raw_results.append(feat)
            
        tracked_results = process_sequence_trajectory(raw_results, fps=200.0)
        return JSONResponse(content={"total_frames": len(tracked_results), "trajectory": tracked_results})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
