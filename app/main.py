from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps
from fastapi.middleware.cors import CORSMiddleware
import io
import torch
from collections import Counter
from .model import load_model  # Make sure this points to your model.py
from .utils import shred_full_page, preprocess_batch # Make sure this points to your utils.py
import logging
import sys
import requests

# --- Setup detailed logging ---
# This will print INFO messages to your console
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
# ---

logger.info("Starting API server...")
try:
    model = load_model()
    logger.info("✅ Model loaded successfully.")
except Exception as e:
    logger.error(f"🔥 FATAL ERROR: Could not load model. API will not work. Error: {e}", exc_info=True)
    model = None # Set model to None so endpoint can fail gracefully

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
    except Exception as e:
        logger.error(f"Image loading error: {e}", exc_info=True)
        raise HTTPException(400, detail=f"Invalid image file. Error: {e}")

    # === Step 2: Shred Image into Patches ===
    try:
        logger.info("Step 2: Shredding full page into patches...")
        
        # UPDATED: Increased max_regions to 100 to allow finding enough patches
        patches = shred_full_page(image, max_regions=100) 
        
        if not patches:
            logger.warning("Shredding returned no patches.")
            raise HTTPException(400, detail="Could not find any clear handwriting on this page.")
        
        # UPDATED: Check for minimum patch count (40)
        if len(patches) < 40:
            logger.warning(f"Insufficient patches found: {len(patches)} (Required: 40)")
            raise HTTPException(
                status_code=400, 
                detail=f"Image is not clear enough (only {len(patches)} text regions found). Please upload a valid, clearer, or fuller page of handwriting."
            )
        
        logger.info(f"Shredding complete. Found {len(patches)} patches.")
    except HTTPException as he:
        raise he # Propagate the HTTP 400 error we just raised
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
        # model.predict_batch returns a list of [('label', confidence_score)]
        predictions = model.predict_batch(batch_tensor)
        logger.info(f"Prediction complete. Raw results: {predictions}")
    except Exception as e:
        logger.error(f"Model prediction error: {e}", exc_info=True)
        raise HTTPException(500, detail=f"Error during model prediction: {e}")

    # === Step 5: Vote and Tally Results ===
    try:
        logger.info("Step 5: Tallying votes...")
        # Only count votes where the model was reasonably confident (e.g., > 60%)
        # This prevents low-confidence guesses from muddying the election
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

        # Tally the confident votes
        votes = [p[0] for p in confident_votes]
        vote_counts = Counter(votes)
        winner, count = vote_counts.most_common(1)[0]
        
        # Consensus score = How many confident patches voted for the winner / TOTAL patches analyzed
        total_patches = len(patches)
        consensus_score = count / total_patches
        
        logger.info(f"Vote complete. Winner: {winner} (Count: {count}). Consensus: {consensus_score:.2f}")

        # --- NEW SMART MESSAGING LOGIC ---
        
        final_label = winner
        
        if winner == "Unknown writer":
            # The model is confident this is NOT a known user.
            final_label = "Unrecognized"
            if consensus_score > 0.5: # Majority of patches agree it's "Unknown"
                final_message = f"High-confidence rejection. This handwriting does not match known users. (Possible impersonator)."
            else: # 'Unknown' won, but it was a close race
                final_message = f"Ambiguous result. No single writer was a clear match. (Possible impersonator)."
        
        else: # A known writer won the vote
            if consensus_score > 0.65: # Strong match
                final_message = f"Strong match for {winner}. (Consensus: {consensus_score*100:.0f}%)"
            elif consensus_score > 0.3: # Weak match (like your 'onome' example)
                final_message = f"Weak match for {winner}. (Consensus: {consensus_score*100:.0f}%). Result is ambiguous."
            else: # Very weak match (e.g., 2 patches out of 50 voted for this person)
                final_label = "Unrecognized"
                final_message = f"Very weak match. Could not confidently identify. (Consensus: {consensus_score*100:.0f}%)"

        if winner in user_to_reg:
            reg_number = user_to_reg[winner]
            headers = {"Authorization": f"Bearer {bearer_token}"}
            try:
                response = requests.get(f"https://api.eceunn.com/api/student/{reg_number}", headers=headers)
                response.raise_for_status()  # Raise an exception for bad status codes
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
                student_info = {"error": "Could not fetch student details."}
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