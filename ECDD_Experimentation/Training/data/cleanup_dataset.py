"""
Dataset Cleanup Script
Removes unused extracted images after dataset split creation.
"""
from pathlib import Path
import shutil


BASE_DIR = Path(r"f:\Team converge\Team-Converge\ECDD_Experimentation")
DATASET_DIR = BASE_DIR / "ECDD_Training_Data"

def cleanup_data():
    """Remove original source data and temporary processing files."""
    
    print("="*60)
    print("DATASET CLEANUP")
    print("="*60)
    
    # Paths to remove
    paths_to_remove = [
        # Source Data
        DATASET_DIR / "ECDD_Datasets" / "Resnet50_Finetune1",
        DATASET_DIR / "ECDD_Datasets" / "Resnet50_Finetune2",
        
        # Temporary Processing Data
        DATASET_DIR / "processed" / "temp_extracted_ft1",
        DATASET_DIR / "processed" / "temp_filtered_ft2",
        DATASET_DIR / "processed" / "extracted" # Old temp dir if exists
    ]
    
    files_removed = 0
    space_freed = 0
    
    for p in paths_to_remove:
        if p.exists():
            print(f"Removing: {p}")
            try:
                # Calculate size before deleting
                size = 0
                for f in p.rglob('*'):
                    if f.is_file():
                        size += f.stat().st_size
                
                shutil.rmtree(p)
                space_freed += size
                print(f"  -> Deleted ({size / (1024*1024):.1f} MB freed)")
            except Exception as e:
                print(f"  -> Error deleting: {e}")
        else:
            print(f"Skipping (not found): {p}")
            
    print("-" * 60)
    print(f"Total space freed: {space_freed / (1024*1024*1024):.2f} GB")
    print("Cleanup Complete.")

if __name__ == "__main__":
    print("WARNING: This will permanently delete original source datasets.")
    print("Only the final 'processed/splits' will remain.")
    # Auto-confirming as per user request "feel free to delete"
    cleanup_data()
