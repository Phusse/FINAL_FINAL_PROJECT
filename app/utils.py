from PIL import Image
import torchvision.transforms as transforms
import cv2
import numpy as np
import math
import torch
import logging

# Setup logger for this file
logger = logging.getLogger(__name__)

# Standard normalization matching your training
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def convert_to_square_patches(img_np):
    """
    Applies the 'Master Shredder' logic to a single numpy image region.
    Returns a LIST of 224x224 PIL Images (1 or more depending on input shape).
    """
    h, w = img_np.shape
    patches = []

    # Case 1: Already mostly square (aspect ratio between 0.5 and 2.0)
    if 0.5 < w/h < 2.0:
        if h > w:
            pad_l = (h - w) // 2
            pad_r = h - w - pad_l
            img_np = cv2.copyMakeBorder(img_np, 0, 0, pad_l, pad_r, cv2.BORDER_CONSTANT, value=[255])
        elif w > h:
            pad_t = (w - h) // 2
            pad_b = w - h - pad_t
            img_np = cv2.copyMakeBorder(img_np, pad_t, pad_b, 0, 0, cv2.BORDER_CONSTANT, value=[255])
            
        resized = cv2.resize(img_np, (224, 224), interpolation=cv2.INTER_AREA)
        return [Image.fromarray(resized)]

    # Case 2: Long strip (needs chopping)
    num_patches = math.ceil(w / h)
    patch_width = math.ceil(w / num_patches)

    for i in range(num_patches):
        start_x = i * patch_width
        end_x = min(start_x + patch_width, w)
        patch = img_np[0:h, start_x:end_x]
        
        # Pad to perfect square
        ph, pw = patch.shape
        if pw < ph:
            pad_l = (ph - pw) // 2
            pad_r = ph - pw - pad_l
            patch = cv2.copyMakeBorder(patch, 0, 0, pad_l, pad_r, cv2.BORDER_CONSTANT, value=[255])
        elif ph < pw:
             pad_t = (pw - ph) // 2
             pad_b = pw - ph - pad_t
             patch = cv2.copyMakeBorder(patch, pad_t, pad_b, 0, 0, cv2.BORDER_CONSTANT, value=[255])

        # Final resize
        final_patch = cv2.resize(patch, (224, 224), interpolation=cv2.INTER_AREA)
        patches.append(Image.fromarray(final_patch))
        
    return patches

def _fallback_grid_shred(full_img_np):
    """
    Dumb but reliable fallback: Chops image into a 3x3 grid.
    This guarantees 9 patches for voting.
    """
    logger.warning("Smart shredder found 0 regions. Using fallback 3x3 grid shredder.")
    h, w = full_img_np.shape
    patch_h, patch_w = h // 3, w // 3
    final_patches = []
    
    for i in range(3): # row
        for j in range(3): # col
            # Crop the grid square
            patch = full_img_np[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
            # Ensure it's valid before processing
            if patch.size == 0: continue
            # Convert this simple crop into a *standardized* 224x224 patch
            # This will pad/squish the grid cell, which is OK for a fallback.
            squares = convert_to_square_patches(patch)
            final_patches.extend(squares)
            
    return final_patches

def shred_full_page(pil_image, max_regions=30):
    """
    Takes a full page, finds text regions, and runs them through the square-ifier.
    """
    full_img_np = np.array(pil_image.convert('L'))

    # 1. Rough segmentation to find text blobs
    _, thresh = cv2.threshold(full_img_np, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5)) 
    dilated = cv2.dilate(thresh, kernel, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_regions]

    final_patches = []
    for c in contours:
        # --- FIX #1: Made shredder more sensitive ---
        if cv2.contourArea(c) < 500: # Was 1000, now 500
            continue
        
        x, y, w, h = cv2.boundingRect(c)
        pad = 15
        x, y = max(0, x-pad), max(0, y-pad)
        w, h = min(full_img_np.shape[1]-x, w+2*pad), min(full_img_np.shape[0]-y, h+2*pad)
        
        region = full_img_np[y:y+h, x:x+w]
        
        squares = convert_to_square_patches(region)
        final_patches.extend(squares)

    # --- FIX #2: Replaced "squish" fallback with "grid" fallback ---
    if not final_patches:
         return _fallback_grid_shred(full_img_np)

    return final_patches

def preprocess_batch(patches: list):
    """Stacks PIL patches into a single PyTorch batch tensor."""
    return torch.stack([transform(p) for p in patches])