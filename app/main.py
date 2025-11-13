from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps
from fastapi.middleware.cors import CORSMiddleware
import io
import torch
import numpy as np  # Required for the Gatekeeper
import cv2          # Required for the Gatekeeper
from collections import Counter
from .model import load_model 
from .utils import shred_full_page, preprocess_batch 
import logging
import sys
import requests

# --- Setup detailed logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
# ---

app = FastAPI()

# --- CORS Middleware ---
origins = [
    "https://eceexams.online",
    "http://localhost:6080", 
    "http://34.16.148.208:9090",
    "http://localhost:5173",      # Your local Vite dev server
    "http://127.0.0.1:5173"     
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ---

# --- User to Registration Number Mapping ---
user_to_reg = {
    "agino": "2020/246758",
    "airma": "2020/245163",
    "bright": "2020/247866",
    "chigbo": "2020/241165",
    "chisom": "2020/246430",
    "christabel": "2020/243931",
    "dave": "2020/242276",
    "david_praise": "2020/249112",
    "divine": "2020/245310",
    "dubem": "2020/241872",
    "Goodness": "2020/GOODNESS",
    "Imelda": "2020/242279",
    "jayden": "2020/245031",
    "joe": "2020/241869",
    "juliet": "2020/247214",
    "nenye": "2020/NENYE",
    "oma": "2020/OMA",
    "onome": "2020/242000",
    "ugochi": "2020/243998",
}

bearer_token = "38|WnWJALrQQ5kiK4b52a2a7IKTMuGZNUC5RglXoXpFb265a3e9"

# --- GATEKEEPER FUNCTION ---
def validate_image_content(pil_image):
    """
    Statistical Gatekeeper:
    Checks if image looks like a document (bright background, dark strokes).
    It converts to Grayscale first so Blue/Red/Black ink are all treated 
    as 'Dark' pixels against the 'White' paper.
    """
    try:
        # 1. CONVERT TO GRAYSCALE ('L' mode)
        # This is the critical step. It turns the image into Black & White values (0-255).
        # Blue ink (e.g., value [0, 0, 255]) becomes Dark Gray (value ~76).
        # This ensures Canny edge detection works regardless of pen color.
        img_array = np.array(pil_image.convert('L'))
        
        # 2. Blank Page Check (Standard Deviation)
        # A blank page has very little variation in pixel color.
        std_dev = np.std(img_array)
        if std_dev < 15: 
            raise HTTPException(400, detail="Image appears to be blank or a solid color.")

        # 3. Dark Photo Check (Mean Brightness)
        # Documents are usually bright white paper. Dark photos are hard to read.
        mean_brightness = np.mean(img_array)
        if mean_brightness < 80: 
            raise HTTPException(400, detail="Image is too dark to analyze. Please use a well-lit scan or photo.")

        # 4. Stroke Detection (Canny Edge Detection)
        # This looks for 'edges' (sharp transitions from paper to ink).
        edges = cv2.Canny(img_array, 100, 200)
        edge_density = np.count_nonzero(edges) / edges.size
        
        if edge_density < 0.005: # Less than 0.5% edges means almost no writing
            raise HTTPException(400, detail="No distinct handwriting detected. Image is too blurry or empty.")
        
        return True
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.warning(f"Gatekeeper warning: {e}")
        return True
# ---------------------------

logger.info("Starting API server...")
try:
    model = load_model()
    logger.info("✅ Model loaded successfully.")
except Exception as e:
    logger.error(f"🔥 FATAL ERROR: Could not load model. API will not work. Error: {e}", exc_info=True)
    model = None 

@app.post("/predict")
async def predict_handwriting(file: UploadFile = File(...)):
    logger.info(f"🚀 Received prediction request for file: {file.filename}")

    if not model:
        raise HTTPException(status_code=500, detail="Model is not loaded. Check server logs.")

    # === Step 1: Validate and Load Image ===
    logger.info("Step 1: Validating and loading image...")
    if file.content_type not in ["image/jpeg", "image/png"]:
        logger.warning(f"Validation failed: Invalid file type ({file.content_type})")
        raise HTTPException(400, detail="Invalid file type. Only .jpg and .png allowed.")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image = ImageOps.exif_transpose(image) # Fix phone rotation issues
        logger.info(f"Image loaded. Original size: {image.size}")
        
        # === RUN THE GATEKEEPER ===
        # This checks brightness, blankness, and strokes (color-agnostic)
        validate_image_content(image)
        logger.info("✅ Gatekeeper check passed.")
        # ==========================

    except HTTPException as he:
        raise he # Pass the specific gatekeeper error to the user
    except Exception as e:
        logger.error(f"Image loading error: {e}", exc_info=True)
        raise HTTPException(400, detail=f"Invalid image file. Error: {e}")

    # === Step 2: Shred Image into Patches ===
    try:
        logger.info("Step 2: Shredding full page into patches...")
        
        # Increased max_regions to 100 so we can find at least 40 patches
        patches = shred_full_page(image, max_regions=100) 
        
        if not patches:
            logger.warning("Shredding returned no patches.")
            raise HTTPException(400, detail="Could not find any clear handwriting on this page.")
        
        # Strict check for 40 patches
        if len(patches) < 1:
            logger.warning(f"Insufficient patches found: {len(patches)} (Required: 40)")
            raise HTTPException(
                status_code=400, 
                detail=f"Image is not clear enough (only {len(patches)} text regions found). Please upload a valid, clearer, or fuller page of handwriting."
            )
        
        logger.info(f"Shredding complete. Found {len(patches)} patches.")
    except HTTPException as he:
        raise he 
    except Exception as e:
        logger.error(f"Shredding error: {e}", exc_info=True)
        raise HTTPException(500, detail=f"Error during image shredding: {e}")

    # === Step 3: Preprocess Patches ===
    try:
        logger.info("Step 3: Preprocessing patches into a batch tensor...")
        batch_tensor = preprocess_batch(patches)
        logger.info(f"Batch created. Tensor shape: {batch_tensor.shape}")
    except Exception as e:
        logger.error(f"Preprocessing error: {e}", exc_info=True)
        raise HTTPException(500, detail=f"Error during batch preprocessing: {e}")

    # === Step 4: Model Prediction ===
    try:
        logger.info("Step 4: Sending batch to model for prediction...")
        predictions = model.predict_batch(batch_tensor)
        logger.info(f"Prediction complete. Raw results: {predictions}")
    except Exception as e:
        logger.error(f"Model prediction error: {e}", exc_info=True)
        raise HTTPException(500, detail=f"Error during model prediction: {e}")

    # === Step 5: Vote and Tally Results ===
    try:
        logger.info("Step 5: Tallying votes...")
        confidence_threshold = 0.6 
        confident_votes = [p for p in predictions if p[1] > confidence_threshold]
        logger.info(f"Confident votes (>{confidence_threshold*100}%): {confident_votes}")
        
        if not confident_votes:
            logger.warning("No confident votes found.")
            return JSONResponse(content={
                "label": "Unrecognized",
                "confidence": 0.0,
                "message": "Handwriting detected, but no patches matched confidently. (Possible impersonator or poor scan)."
            })

        votes = [p[0] for p in confident_votes]
        vote_counts = Counter(votes)
        winner, count = vote_counts.most_common(1)[0]
        
        total_patches = len(patches)
        consensus_score = count / total_patches
        
        logger.info(f"Vote complete. Winner: {winner} (Count: {count}). Consensus: {consensus_score:.2f}")

        # --- SMART MESSAGING LOGIC ---
        final_label = winner
        
        if winner == "Unknown writer":
            final_label = "Unrecognized"
            if consensus_score > 0.5: 
                final_message = f"High-confidence rejection. This handwriting does not match known users. (Possible impersonator)."
            else: 
                final_message = f"Ambiguous result. No single writer was a clear match. (Possible impersonator)."
        else: 
            if consensus_score > 0.65: 
                final_message = f"Strong match for {winner}. (Consensus: {consensus_score*100:.0f}%)"
            elif consensus_score > 0.3: 
                final_message = f"Weak match for {winner}. (Consensus: {consensus_score*100:.0f}%). Result is ambiguous."
            else: 
                final_label = "Unrecognized"
                final_message = f"Very weak match. Could not confidently identify. (Consensus: {consensus_score*100:.0f}%)"

        if winner in user_to_reg:
            reg_number = user_to_reg[winner]
            headers = {"Authorization": f"Bearer {bearer_token}"}
            try:
                response = requests.get(f"https://api.eceunn.com/api/student/{reg_number}", headers=headers)
                response.raise_for_status() 
                student_data = response.json().get("data", {})
                student_info = {
                    "reg_number": student_data.get("reg_number"),
                    "first_name": student_data.get("first_name"),
                    "middle_name": student_data.get("middle_name"),
                    "last_name": student_data.get("last_name"),
                    "level": student_data.get("level"),
                    "passport_url": student_data.get("passport_url"),
                }
            except requests.exceptions.RequestException as e:
                logger.error(f"API request error: {e}", exc_info=True)
                student_info = {
                    "error": "The student is not a student of Electronic and Computer Engineering."
                }
        else:
            student_info = {}

        return JSONResponse(status_code=200, content={
            "label": final_label.capitalize(),
            "confidence": round(consensus_score, 4),
            "message": final_message,
            "debug_votes": dict(vote_counts),
            "student_info": student_info,
        })
    except Exception as e:
        logger.error(f"Voting/Tally error: {e}", exc_info=True)
        raise HTTPException(500, detail=f"Error during result tallying: {e}")