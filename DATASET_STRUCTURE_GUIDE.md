# Dataset Structure Guide for ECDD LaDeDa Training

## 📁 Required Dataset Structure

Your dataset must be organized in the following structure:

```
ecdd-training-data/
├── train/
│   ├── real/
│   │   ├── real_001.jpg
│   │   ├── real_002.jpg
│   │   └── ... (more real images)
│   └── fake/
│       ├── fake_001.jpg
│       ├── fake_002.jpg
│       └── ... (more fake images)
├── val/
│   ├── real/
│   │   └── ... (validation real images)
│   └── fake/
│       └── ... (validation fake images)
└── test/
    ├── real/
    │   └── ... (test real images)
    └── fake/
        └── ... (test fake images)
```

## 📊 Recommended Dataset Split

- **Train**: 70-80% of total data
- **Validation**: 10-15% of total data
- **Test**: 10-15% of total data

### Example counts:
- Train: 1000-5000+ images per class (real/fake)
- Val: 200-500 images per class
- Test: 200-500 images per class

## 🖼️ Image Requirements

### Supported Formats:
- `.jpg`, `.jpeg`, `.png`
- RGB color images

### Image Content:
- **Must contain faces** (the model is trained for facial deepfake detection)
- Images should be at least 256x256 pixels (will be resized to 256x256)
- Face should be reasonably visible (not too small or occluded)

### Quality Guidelines:
- Avoid extremely low-resolution images (< 128x128)
- JPEG quality should be reasonable (> 30)
- Images can have EXIF orientation metadata (automatically handled)

## 🚀 How to Upload to Google Colab

### Option 1: Direct Upload (Small datasets < 500MB)
1. In Colab, click the **Files** icon on the left sidebar
2. Create folder structure: `ecdd-training-data/train/real`, etc.
3. Upload images to respective folders
4. Update notebook path: `DATA_PATH = "/content/ecdd-training-data"`

### Option 2: Google Drive (Recommended for larger datasets)
1. Upload your dataset to Google Drive with the correct structure
2. Mount Google Drive in Colab:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Update notebook path: `DATA_PATH = "/content/drive/MyDrive/ecdd-training-data"`

### Option 3: Kaggle Datasets (Best for Kaggle notebooks)
1. Go to [kaggle.com](https://www.kaggle.com) → Your Work → Datasets → New Dataset
2. Upload your dataset folder with the correct structure
3. Make the dataset public or private
4. In the Kaggle notebook: Add Data → Your Datasets → Select your dataset
5. Update path: `DATA_PATH = "/kaggle/input/your-dataset-name"`

### Option 4: Download from URL
```python
# Add this cell at the beginning of the notebook
import urllib.request
import zipfile

# Download dataset
url = "https://your-url.com/dataset.zip"
urllib.request.urlretrieve(url, "dataset.zip")

# Extract
with zipfile.ZipFile("dataset.zip", 'r') as zip_ref:
    zip_ref.extractall("/content")

DATA_PATH = "/content/ecdd-training-data"
```

## 📝 Dataset Preparation Tips

### If you have raw videos:
1. Extract frames from videos at 1-5 FPS
2. Use face detection to crop face regions
3. Organize into train/val/test splits

### If you have existing datasets:
Popular deepfake datasets you can use:
- **Celeb-DF v2**: Celebrity deepfakes
- **FaceForensics++**: Multiple manipulation methods
- **DFDC**: Facebook Deepfake Detection Challenge
- **WildDeepfake**: In-the-wild deepfakes

### Balancing your dataset:
- Keep roughly equal numbers of real and fake images
- If imbalanced, the model may be biased toward the majority class

## 🔍 Verify Your Dataset

Before training, run this verification script:

```python
from pathlib import Path

DATA_PATH = "/path/to/your/ecdd-training-data"

for split in ['train', 'val', 'test']:
    split_path = Path(DATA_PATH) / split
    
    # Check if split exists
    if not split_path.exists():
        print(f"❌ Missing: {split}/")
        continue
    
    # Count images
    real_images = list((split_path / "real").glob("*.jpg")) + \
                  list((split_path / "real").glob("*.jpeg")) + \
                  list((split_path / "real").glob("*.png"))
    fake_images = list((split_path / "fake").glob("*.jpg")) + \
                  list((split_path / "fake").glob("*.jpeg")) + \
                  list((split_path / "fake").glob("*.png"))
    
    print(f"✅ {split}:")
    print(f"   Real: {len(real_images)} images")
    print(f"   Fake: {len(fake_images)} images")
    print(f"   Total: {len(real_images) + len(fake_images)} images")
    print()

print("Dataset verification complete!")
```

## ⚙️ Notebook Configuration

In the notebook, update this section (Cell under "2. Dataset Setup"):

```python
# ========== CONFIGURE YOUR DATASET PATH HERE ==========
# For Google Colab with direct upload:
DATA_PATH = "/content/ecdd-training-data"

# For Google Drive:
# DATA_PATH = "/content/drive/MyDrive/ecdd-training-data"

# For Kaggle:
# DATA_PATH = "/kaggle/input/your-dataset-name"
```

## 🎯 Expected Training Results

With a properly structured dataset:
- **Training time**: 30-45 minutes on Colab T4 GPU (15 epochs)
- **Expected accuracy**: 85-95% on test set (depends on dataset quality)
- **Model size**: ~100MB (PyTorch .pth file)

## 🆘 Troubleshooting

### "Dataset not found" error:
- Check that `DATA_PATH` is correct
- Verify folder structure matches exactly (train/val/test → real/fake)
- Make sure images have correct extensions (.jpg, .jpeg, .png)

### "No images found" error:
- Check image file extensions
- Ensure images are directly in real/ or fake/ folders (not in subfolders)

### Out of Memory (OOM) errors:
- Reduce `batch_size` in CONFIG (try 8 or 4 instead of 16)
- Reduce image count in training set
- Restart runtime to clear memory

## 📚 Additional Resources

- **Sample dataset**: Check `ECDD_Experimentation/ECDD_Training_Data/processed/splits/finetune1/` in the repository for example structure
- **Training guide**: See `ECDD_Experimentation/Training/KAGGLE_TRAINING_GUIDE.md`

---

**Ready to train?** Upload your dataset, open `ECDD_LaDeDa_Training_Colab.ipynb` in Google Colab, and run all cells! 🚀
