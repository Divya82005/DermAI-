
from pathlib import Path
import csv

# Dataset paths
DATA_DIR = Path("data/ISIC2017")

IMAGE_DIR = DATA_DIR / "Test_Data"
MASK_DIR = DATA_DIR / "Test_GroundTruth"
CSV_FILE = DATA_DIR / "Test_Diagnosis.csv"

# Check folders and CSV
for path in [IMAGE_DIR, MASK_DIR, CSV_FILE]:
    print(f"{path}: {'FOUND' if path.exists() else 'NOT FOUND'}")

if not IMAGE_DIR.is_dir() or not MASK_DIR.is_dir() or not CSV_FILE.is_file():
    print("\nCheck the folder names and paths before continuing.")
    raise SystemExit(1)

# Count images and masks
images = {
    p.stem: p
    for p in IMAGE_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}
}

masks = {
    p.name.removesuffix("_segmentation.png"): p
    for p in MASK_DIR.iterdir()
    if p.is_file() and p.name.lower().endswith("_segmentation.png")
}

# Read diagnosis CSV
with CSV_FILE.open("r", newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

csv_ids = {row["image_id"].strip() for row in rows}

print("\n--- Dataset summary ---")
print("Test images:", len(images))
print("Segmentation masks:", len(masks))
print("Diagnosis rows:", len(rows))

print("\n--- Matching checks ---")
print("Images without masks:", len(images.keys() - masks.keys()))
print("Masks without images:", len(masks.keys() - images.keys()))
print("Images without diagnosis:", len(images.keys() - csv_ids))
print("Diagnosis IDs without images:", len(csv_ids - images.keys()))

print("\n--- Diagnosis columns ---")
print(reader.fieldnames if rows else "No diagnosis rows found")
