# Dataset Download Guide

This guide provides direct download links and setup instructions for all supported datasets.

---

## Directory Structure Expected

Place datasets inside the `data/` folder of the project:

```
Computer Vision Project/
└── data/
    ├── images/              ← HAM10000 images (.jpg)
    ├── ISIC_2024_metadata.csv  ← HAM10000 CSV
    ├── masks/               ← HAM10000 segmentation masks (optional)
    ├── isic_2019/           ← ISIC 2019
    ├── isic_2020/           ← ISIC 2020
    ├── isic_2024/           ← ISIC 2024
    └── ph2/                 ← PH2
```

---

## Dataset 1: HAM10000 (already supported)

- **Link**: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
- **Size**: ~3.5 GB, 10,015 images
- **Setup**:
  1. Download and extract
  2. Place all `.jpg` images in `data/images/`
  3. Place `HAM10000_metadata.csv` as `data/ISIC_2024_metadata.csv`
  4. (Optional) Place segmentation masks in `data/masks/`

---

## Dataset 2: ISIC 2019

- **Link**: https://challenge.isic-archive.com/data/#2019
- **Direct Kaggle**: https://www.kaggle.com/datasets/andrewmvd/isic-2019
- **Size**: ~9.1 GB, 25,331 images, 8 classes
- **Setup**:
  ```
  data/isic_2019/
  ├── ISIC_2019_Training_Input/     ← all images (ISIC_0xxxxxxx.jpg)
  └── ISIC_2019_Training_GroundTruth.csv
  ```

---

## Dataset 3: ISIC 2020

- **Link**: https://www.kaggle.com/competitions/siim-isic-melanoma-classification/data
- **Size**: ~38 GB (full), 33,126 images; binary (melanoma/benign)
- **Setup**:
  ```
  data/isic_2020/
  ├── train/        ← images (ISIC_xxxxxxxxxx.jpg)
  └── train.csv     ← columns: image_name, patient_id, target (0/1)
  ```
- **Tip**: Download the 768×768 JPEG version (smaller, sufficient quality).

---

## Dataset 4: ISIC 2024 (SLICE-3D)

- **Link**: https://www.kaggle.com/competitions/isic-2024-challenge/data
- **Size**: ~1.5 GB (compressed crops), ~400K images; binary (melanoma/benign)
- **Note**: The code automatically subsamples negatives (3× positives) to handle extreme imbalance.
- **Setup**:
  ```
  data/isic_2024/
  ├── train-image/image/    ← all images (xxxxx.jpg)
  └── train-metadata.csv    ← columns: isic_id, target (0/1)
  ```

---

## Dataset 5: PH2

- **Link**: https://www.fc.up.pt/addi/ph2%20database.html
- **Direct Download**: https://www.dropbox.com/s/k88qukc20ljnbuo/PH2Dataset.rar
- **Size**: ~55 MB, 200 dermoscopy images, 3 classes
- **Setup**:
  ```
  data/ph2/
  ├── PH2Dataset/
  │   ├── IMD001/
  │   │   ├── IMD001_Dermoscopic_Image/IMD001.bmp
  │   │   └── IMD001_lesion/IMD001_lesion.bmp
  │   └── ... (IMD002–IMD200)
  └── PH2_dataset.txt
  ```

---

## Quick Kaggle Download (CLI)

If you have the Kaggle CLI installed:

```bash
# HAM10000
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p data/

# ISIC 2019
kaggle datasets download -d andrewmvd/isic-2019 -p data/isic_2019/

# ISIC 2020
kaggle competitions download -c siim-isic-melanoma-classification -p data/isic_2020/

# ISIC 2024
kaggle competitions download -c isic-2024-challenge -p data/isic_2024/
```

---

## Notes

- **Missing datasets are silently skipped** — the project runs with whatever is available.
- After adding new datasets, simply re-run `python main.py` — no code changes needed.
- For Google Colab: mount Google Drive and update the `DATA_DIR` path in `configs/config.py`.
