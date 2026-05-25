import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix
from skimage.feature import hog
import joblib
import sys
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
META_FILES = {
    "COVID": "COVID.metadata.xlsx",
    "Normal": "Normal.metadata.xlsx",
    "Lung_Opacity": "Lung_Opacity.metadata.xlsx",
    "Viral_Pneumonia": "Viral Pneumonia.metadata.xlsx"
}
IMAGES_SUBFOLDER = "images"
TARGET_SIZE = (224, 224)

def try_read_excel(path):
    try:
        return pd.read_excel(path)
    except ImportError as e:
        print("Error reading Excel file. Missing optional dependency (openpyxl).")
        print("Please run: pip install openpyxl")
        raise e

def detect_filename_column(df):
    candidates = ['filename','file','image','image_name','imagefile','image_path','path','filepath','name','file_name','FileName']
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in cols_lower:
            return cols_lower[cand]
    for c in df.columns:
        sample_vals = df[c].astype(str).head(200).str.lower()
        if sample_vals.str.contains(r'\.png|\.jpg|\.jpeg|\.bmp', regex=True).any():
            return c
    for c in df.columns:
        sample = df[c].astype(str).head(200)
        if sample.str.match(r'^\d{3,6}$').sum() > 10:
            return c
    return None

def build_full_path(entry, label):
    if pd.isna(entry):
        return None
    s = str(entry).strip()
    if s == "":
        return None
    exts = ["", ".png", ".jpg", ".jpeg"]
    candidates = []
    bases = [s, s.replace(" ", ""), s.replace(" ", ""), s.replace("-", "")]
    bases = list(dict.fromkeys(bases))
    for b in bases:
        for ext in exts:
            fname = b + ext
            if IMAGES_SUBFOLDER:
                candidates.append(os.path.join(BASE_DIR, label, IMAGES_SUBFOLDER, fname))
            candidates.append(os.path.join(BASE_DIR, label, fname))
            candidates.append(os.path.join(BASE_DIR, fname))
            candidates.append(os.path.join(BASE_DIR, os.path.normpath(fname)))
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    if IMAGES_SUBFOLDER:
        return os.path.join(BASE_DIR, label, IMAGES_SUBFOLDER, s + ".png")
    return os.path.join(BASE_DIR, label, s + ".png")

dfs = []
for label, fname in META_FILES.items():
    meta_path = os.path.join(BASE_DIR, fname)
    if not os.path.exists(meta_path):
        print(f"[ERROR] Metadata file not found: {meta_path}")
        sys.exit(1)
    print(f"\nLoading metadata for label='{label}' from: {meta_path}")
    df = try_read_excel(meta_path)
    print("Columns detected:", list(df.columns))
    df['label'] = label
    if 'FILE NAME' in df.columns:
        fname_col = 'FILE NAME'
    else:
        fname_col = detect_filename_column(df)
        if fname_col is None:
            print(f"[ERROR] Could not detect filename column in {fname}.")
            print("Columns available:", list(df.columns))
            sys.exit(1)
    print(f"Using column '{fname_col}' for image filenames.")
    df['path'] = df[fname_col].apply(lambda x: build_full_path(x, label))
    df['exists'] = df['path'].apply(lambda p: os.path.exists(p) if isinstance(p, str) else False)
    n_total = len(df)
    n_exist = df['exists'].sum()
    print(f"Found {n_exist}/{n_total} files exist for label '{label}'.")
    if n_exist < max(1, int(0.5 * n_total)):
        print(f"[WARN] Less than 50% of paths exist for label '{label}'. Check IMAGES_SUBFOLDER or filename format.")
        missing_sample = df.loc[~df['exists'], fname_col].astype(str).head(10).tolist()
        print("Sample missing filename entries:", missing_sample)
    dfs.append(df[['path', 'label', 'exists']])

combined = pd.concat(dfs, ignore_index=True)
if 'exists' in combined.columns:
    combined_existing = combined[combined['exists']].reset_index(drop=True)
else:
    combined_existing = combined
print("\nTotal images (existing files):", len(combined_existing))
print("Per-class counts (existing):\n", combined_existing['label'].value_counts())
if len(combined_existing) == 0:
    print("[ERROR] No image files found. Fix metadata paths or move images into expected folders.")
    sys.exit(1)

def preprocess_image(path, size=TARGET_SIZE):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    im = cv2.resize(im, size, interpolation=cv2.INTER_AREA)
    im = im.astype('float32') / 255.0
    return im

def extract_hog_features(df, max_samples=None):
    feats, labels = [], []
    skipped = 0
    for i, row in df.iterrows():
        if max_samples and len(feats) >= max_samples:
            break
        p = row['path']
        im = preprocess_image(p)
        if im is None:
            skipped += 1
            continue
        h = hog(im, pixels_per_cell=(16,16), cells_per_block=(2,2), feature_vector=True)
        feats.append(h)
        labels.append(row['label'])
    if skipped:
        print(f"[INFO] Skipped {skipped} images that couldn't be read.")
    return np.array(feats), np.array(labels)

train_df, test_df = train_test_split(combined_existing[['path','label']], test_size=0.3, stratify=combined_existing['label'], random_state=42)
print(f"\nTrain / Test sizes: {len(train_df)} / {len(test_df)}")
print("\nExtracting HOG features (quick test subset)...")
X_train, y_train = extract_hog_features(train_df, max_samples=500)
X_test, y_test = extract_hog_features(test_df, max_samples=200)
print("Feature shapes:", X_train.shape, X_test.shape)
if X_train.size == 0 or X_test.size == 0:
    print("[ERROR] Feature extraction returned empty arrays. Check image readability and paths.")
    sys.exit(1)

model = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=3, gamma='scale', class_weight='balanced', probability=True))
print("\nTraining SVM...")
model.fit(X_train, y_train)
print("\nEvaluating on test subset...")
y_pred = model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
classes = model.named_steps['svc'].classes_ if 'svc' in model.named_steps else np.unique(y_test)
fig, ax = plt.subplots(figsize=(6,6))
im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_xticks(np.arange(len(classes)))
ax.set_yticks(np.arange(len(classes)))
ax.set_xticklabels(classes, rotation=45, ha='right')
ax.set_yticklabels(classes)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
thresh = cm.max() / 2.0
total = cm.sum()
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        val = cm[i, j]
        pct = val / total if total > 0 else 0
        color = "white" if val > thresh else "black"
        ax.text(j, i, f"{val:d}\n{pct:.2%}", ha="center", va="center", color=color, fontsize=9)
plt.tight_layout()
models_dir = os.path.join(BASE_DIR, "models")
os.makedirs(models_dir, exist_ok=True)
plt.savefig(os.path.join(models_dir, "confusion_matrix.png"), dpi=150, bbox_inches='tight')
plt.show()
joblib.dump(model, os.path.join(models_dir, "svm_hog.pkl"))
print("\nModel saved to:", os.path.join(models_dir, "svm_hog.pkl"))
