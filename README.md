# COVID-19 Chest X-Ray Classifier

A classical machine learning pipeline that classifies chest X-ray images into **COVID-19**, **Lung Opacity**, and **Normal** categories using HOG feature extraction and a Support Vector Machine (SVM).

---

## Dataset

**[COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)** (Kaggle)

| Class | Images |
|---|---|
| Normal | 10,192 |
| Lung_Opacity | 6,012 |
| COVID | 3,616 |
| **Total** | **19,820** |

---

## Project Structure

```
project-root/
│
├── COVID/
│   ├── images/
│   └── COVID.metadata.xlsx
│
├── Normal/
│   ├── images/
│   └── Normal.metadata.xlsx
│
├── Lung_Opacity/
│   ├── images/
│   └── Lung_Opacity.metadata.xlsx
│
├── Viral_Pneumonia/
│   ├── images/
│   └── Viral Pneumonia.metadata.xlsx
│
├── models/
│   ├── svm_hog.pkl
│   └── confusion_matrix.png
│
└── covid_detection.py
```

---

## Requirements

```bash
pip install opencv-python numpy pandas scikit-learn scikit-image joblib matplotlib openpyxl
```

---

## How It Works

1. **Metadata Loading** — Reads `.metadata.xlsx` files to resolve image paths per class
2. **Preprocessing** — Grayscale conversion, resize to `224×224`, normalize to `[0, 1]`
3. **HOG Features** — Extracts shape-based features (16×16 px cells, 2×2 blocks) → 6,084-dim vector
4. **SVM Classifier** — RBF kernel, `C=3`, `gamma='scale'`, balanced class weights inside a `StandardScaler` pipeline
5. **Evaluation** — Classification report + confusion matrix saved to `models/`

---

## Results

> Trained on 500 samples, evaluated on 200 samples (quick-run subset).

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| COVID | 0.88 | 0.41 | 0.56 | 37 |
| Lung_Opacity | 0.68 | 0.80 | 0.74 | 65 |
| Normal | 0.79 | 0.87 | 0.83 | 98 |
| **Accuracy** | | | **0.76** | **200** |
| Macro Avg | 0.79 | 0.69 | 0.71 | 200 |
| Weighted Avg | 0.77 | 0.76 | 0.75 | 200 |

### Confusion Matrix

![Confusion Matrix](models/confusion_matrix.png)

**Key observations:**
- **Normal** class performs best (F1: 0.83) — highest sample count helps
- **COVID** has high precision (0.88) but low recall (0.41) — many COVID cases misclassified as Lung_Opacity or Normal
- **Lung_Opacity** shows balanced performance (F1: 0.74)

---

## Usage

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) and place folders in the project root.

2. Run the script:
```bash
python covid_detection.py
```

3. Outputs:
   - `models/svm_hog.pkl` — saved model
   - `models/confusion_matrix.png` — confusion matrix plot

---

## Configuration

| Constant | Default | Description |
|---|---|---|
| `TARGET_SIZE` | `(224, 224)` | Image resize dimensions |
| `max_samples` (train) | `500` | Training sample cap — remove for full training |
| `max_samples` (test) | `200` | Test sample cap — remove for full evaluation |
| `test_size` | `0.3` | Train/test split ratio |

> To train on the **full dataset** (13,874 train / 5,946 test), remove the `max_samples` argument in `extract_hog_features()` calls.

---

## Limitations

- Current results are from a **capped 500-sample subset**; full-dataset training will improve recall significantly, especially for COVID.
- HOG + SVM is a classical approach. For production-grade accuracy, consider CNN-based transfer learning (ResNet, EfficientNet).
- The dataset is **imbalanced** (Normal >> COVID); the `class_weight='balanced'` parameter partially compensates for this.

---
