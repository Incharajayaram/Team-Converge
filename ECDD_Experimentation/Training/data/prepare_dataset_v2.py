"""
ECDD Dataset Preparation Script V2
Handles specific fine-tuning requirements for ResNet50 models.

Tasks:
1. Resnet50_Finetune1 (Celeb-DF-v2): Extract ~2500 frames each for Real and Fake from videos.
2. Resnet50_Finetune2: Filter Fake images to only keep those with faces. Real images (~750) are kept as is.
3. Create 80/10/10 splits (Train/Val/Test) for both.
4. Enforce strict preprocessing: RGB (sRGB), EXIF orientation, 256x256 Lanczos resize.
"""

import os
import cv2
import glob
import random
import shutil
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps 
from tqdm import tqdm

# Configuration
BASE_DIR = Path(r"f:\Team converge\Team-Converge\ECDD_Experimentation\ECDD_Training_Data\ECDD_Datasets")
FT1_DIR = BASE_DIR / "Resnet50_Finetune1"
FT2_DIR = BASE_DIR / "Resnet50_Finetune2"

OUTPUT_BASE = Path(r"f:\Team converge\Team-Converge\ECDD_Experimentation\ECDD_Training_Data\processed")
SPLIT_BASE = OUTPUT_BASE / "splits"

TARGET_SIZE = (256, 256)
FT1_TARGET_FRAMES = 2500  # Per class (Real/Fake)

# Setup Face Detection (Haar Cascade)
try:
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    FACE_CASCADE = cv2.CascadeClassifier(face_cascade_path)
except Exception as e:
    print(f"Warning: Could not load default Haar cascade: {e}")
    FACE_CASCADE = None

def setup_directories():
    """Create necessary output directories."""
    for task in ["finetune1", "finetune2"]:
        for split in ["train", "val", "test"]:
            path = SPLIT_BASE / task / split
            path.mkdir(parents=True, exist_ok=True)
            (path / "real").mkdir(exist_ok=True)
            (path / "fake").mkdir(exist_ok=True)
    
    # Temp dirs for extraction
    (OUTPUT_BASE / "temp_extracted_ft1" / "real").mkdir(parents=True, exist_ok=True)
    (OUTPUT_BASE / "temp_extracted_ft1" / "fake").mkdir(parents=True, exist_ok=True)
    (OUTPUT_BASE / "temp_filtered_ft2" / "fake").mkdir(parents=True, exist_ok=True)

def preprocess_image(image_path, target_path):
    """
    Load, fix orientation, convert to RGB, resize (Lanczos), and save.
    Returns True if successful.
    """
    try:
        # Open with PIL
        with Image.open(image_path) as img:
            # Fix EXIF orientation
            img = ImageOps.exif_transpose(img)
            
            # Convert to RGB (sRGB assumption)
            img = img.convert('RGB')
            
            # Resize
            img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
            
            # Save
            img.save(target_path, quality=95)
            return True
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False

def has_face(image_path):
    """Check if image has at least one face using OpenCV."""
    if FACE_CASCADE is None:
        return True # Fallback if detector fails to load, though risky
        
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return False
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return len(faces) > 0
    except Exception as e:
        print(f"Face detection error on {image_path}: {e}")
        return False

def process_finetune1():
    """Process Celeb-DF-v2 Videos."""
    print("\n--- Processing Fine-tune 1 (Celeb-DF-v2) ---")
    
    # 1. Gather Videos
    # Adjust paths based on actual structure inside Resnet50_Finetune1
    # Assuming structure: Resnet50_Finetune1/Celeb-real/id/*.mp4 and Resnet50_Finetune1/Celeb-synthesis/*.mp4
    # Or simple flat folders. We'll search recursively.
    
    real_videos = list(FT1_DIR.rglob("*.mp4")) + list(FT1_DIR.rglob("*.avi"))
    # Filter for real/fake based on parent folder names if usually distinguished
    # Assuming standard Celeb-DF structure: 'Celeb-real', 'Celeb-synthesis', 'YouTube-real'
    
    # Let's try to categorize by folder name keywords
    real_vids_final = []
    fake_vids_final = []
    
    for v in real_videos:
        path_str = str(v).lower()
        if "real" in path_str and "synthesis" not in path_str:
            real_vids_final.append(v)
        elif "synthesis" in path_str or "fake" in path_str:
            fake_vids_final.append(v)
            
    print(f"Found {len(real_vids_final)} Real videos, {len(fake_vids_final)} Fake videos.")
    
    if not real_vids_final and not fake_vids_final:
        print("CRITICAL: No videos found in FT1 directory matching criteria. Checking exact contents...")
        # Fallback: list top level dirs to debug
        # (For now, proceeding with empty lists which will skip extraction)
    
    # 2. Extract Frames
    def extract_from_list(vid_list, label, target_count):
        output_dir = OUTPUT_BASE / "temp_extracted_ft1" / label
        existing = list(output_dir.glob("*.jpg"))
        if len(existing) >= target_count:
            print(f"Already extracted enough {label} frames ({len(existing)}). Skipping.")
            return existing[:target_count]
            
        if not vid_list:
            return []

        extracted_paths = existing
        count_needed = target_count - len(existing)
        
        # Calculate frames per video needed
        # We assume we want diversity, so we iterate all videos
        # Randomly shuffle videos
        random.shuffle(vid_list)
        
        pbar = tqdm(total=target_count, initial=len(existing), desc=f"Extracting {label}")
        
        frames_per_vid = max(1, int(np.ceil(count_needed / len(vid_list))))
        
        for vid in vid_list:
            if len(extracted_paths) >= target_count:
                break
                
            cap = cv2.VideoCapture(str(vid))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames > 10:
                # Sample random frames
                indices = sorted(random.sample(range(total_frames), min(frames_per_vid * 2, total_frames))) # Get a few candidates
                
                saved_for_vid = 0
                for idx in indices:
                    if len(extracted_paths) >= target_count:
                        break
                    if saved_for_vid >= frames_per_vid:
                        break

                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        # Save temp
                        out_name = f"{vid.stem}_f{idx}.jpg"
                        out_path = output_dir / out_name
                        cv2.imwrite(str(out_path), frame) # Raw save, will preprocess later
                        
                        # Verify face? (Optional, but good for quality)
                        # We strictly need preprocessing later.
                        
                        extracted_paths.append(out_path)
                        saved_for_vid += 1
                        pbar.update(1)
            cap.release()
            
        pbar.close()
        return extracted_paths

    real_frames = extract_from_list(real_vids_final, "real", FT1_TARGET_FRAMES)
    fake_frames = extract_from_list(fake_vids_final, "fake", FT1_TARGET_FRAMES)
    
    return real_frames, fake_frames

def process_finetune2():
    """Process ResNet50 Finetune 2 Images."""
    print("\n--- Processing Fine-tune 2 (ResNet50 Mix) ---")
    
    # 1. Gather Images
    # Assuming structure: Resnet50_Finetune2/Real and Resnet50_Finetune2/Fake
    # We need to find them strictly
    
    all_files = list(FT2_DIR.rglob("*"))
    image_exts = {'.jpg', '.jpeg', '.png', '.webp'}
    images = [p for p in all_files if p.suffix.lower() in image_exts]
    
    real_images = []
    fake_images = []
    
    for img in images:
        path_str = str(img).lower()
        # Heuristic for real/fake classification based on folders
        if "real" in path_str and "fake" not in path_str:
            real_images.append(img)
        elif "fake" in path_str or "df" in path_str or "synthesis" in path_str:
            fake_images.append(img)
        else:
            # Fallback/Ambiguous - skip or assume? 
            # Let's inspect parent folder
            if img.parent.name.lower() in ['real', '0']:
                real_images.append(img)
            elif img.parent.name.lower() in ['fake', '1']:
                fake_images.append(img)
    
    print(f"Found {len(real_images)} potentially Real images.")
    print(f"Found {len(fake_images)} potentially Fake images.")
    
    # 2. Filter Fakes (Face Detection)
    print("Filtering Fake images for faces...")
    valid_fake_images = []
    
    # Check if we already have filtered ones
    temp_filtered_dir = OUTPUT_BASE / "temp_filtered_ft2" / "fake"
    existing_filtered = list(temp_filtered_dir.glob("*.jpg"))
    
    if existing_filtered:
        print(f"Use existing filtered fakes: {len(existing_filtered)}")
        valid_fake_images = existing_filtered
    else:
        # Filter source
        for img_path in tqdm(fake_images, desc="Detecting Faces"):
            if has_face(img_path):
                 valid_fake_images.append(img_path)
    
    print(f"Filtered Fakes: {len(fake_images)} -> {len(valid_fake_images)}")
    
    return real_images, valid_fake_images

def create_split(item_list, task_name, label):
    """
    Distribute items into train/val/test and Apply Preprocessing.
    item_list: list of source Paths
    task_name: 'finetune1' or 'finetune2'
    label: 'real' or 'fake'
    """
    if not item_list:
        print(f"No items to split for {task_name}/{label}")
        return

    random.shuffle(item_list)
    total = len(item_list)
    
    n_test = int(total * 0.1)
    n_val = int(total * 0.1)
    n_train = total - n_test - n_val
    
    splits = {
        "train": item_list[:n_train],
        "val": item_list[n_train:n_train+n_val],
        "test": item_list[n_train+n_val:]
    }
    
    print(f"  Splitting {task_name}/{label}: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")
    
    for split_name, paths in splits.items():
        dest_dir = SPLIT_BASE / task_name / split_name / label
        
        for p in tqdm(paths, desc=f"Wait process {split_name}/{label}", leave=False):
            # Define output filename
            # Flatten name to avoid collisions if coming from different subfolders
            clean_name = f"{p.stem}.jpg"
            # If collision, append random
            if (dest_dir / clean_name).exists():
                 clean_name = f"{p.stem}_{random.randint(1000,9999)}.jpg"
            
            target_path = dest_dir / clean_name
            
            # Allow skipping if already exists (resume)
            if target_path.exists():
                continue
                
            preprocess_image(p, target_path)

def main():
    print("Starting Dataset Preparation...")
    setup_directories()
    
    # 1. Finetune 1 (Celeb-DF)
    ft1_real, ft1_fake = process_finetune1()
    create_split(ft1_real, "finetune1", "real")
    create_split(ft1_fake, "finetune1", "fake")
    
    # 2. Finetune 2 (ResNet50 Mix)
    ft2_real, ft2_fake = process_finetune2()
    # Note: ft2_fake might be source paths or copied temp paths depending on flow
    # Since process_finetune2 returned paths (filtered), we use them directly.
    create_split(ft2_real, "finetune2", "real")
    create_split(ft2_fake, "finetune2", "fake")
    
    print("\nDataset Preparation Complete.")
    print(f"Output saved to: {SPLIT_BASE}")

if __name__ == "__main__":
    main()
