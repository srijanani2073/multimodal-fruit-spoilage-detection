"""
Export one sRGB + one thermal sample image for every TR-6 class
(6 produce types x {not spoiled, spoiled}) into the website's
assets/dataset/ folder.

Run once from the website folder on the machine that has the dataset:

    python tools/export_dataset_samples.py --root ~/Desktop/Thesis/TR-6

For each class it takes the middle image of the sorted sRGB folder and the
middle image of the sorted thermal (IR fusion) folder, so the samples are
representative of the trajectory rather than its first or last day.
Existing images in assets/dataset/ are overwritten.
"""
import argparse
from pathlib import Path

from PIL import Image

FRUITS = ["Banana", "Carrot", "Guava", "Indian_Gooseberry", "Mango", "Tomato"]
LABELS = {"Not_spoiled": "notspoiled", "Spoiled": "spoiled"}
IR_FOLDER_NAMES = ["IR_fusion_images", "IR_Fusion_images", "ir_fusion_images"]
MAX_SIDE = 900


def find_ir_folder(base: Path):
    for name in IR_FOLDER_NAMES:
        if (base / name).exists():
            return base / name
    return None


def images_in(folder):
    if folder is None or not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def save_middle(images, out_path: Path):
    if not images:
        return False
    src = images[len(images) // 2]
    img = Image.open(src).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    img.save(out_path, quality=86, optimize=True)
    print(f"  {out_path.name:<40} <- {src.name}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="~/Desktop/Thesis/TR-6", help="TR-6 dataset root")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "assets" / "dataset"))
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    missing = []

    for fruit in FRUITS:
        for folder_label, slug_label in LABELS.items():
            base = root / "Classified" / fruit / folder_label
            slug = f"{fruit.lower()}_{slug_label}"
            print(f"{fruit} / {folder_label}")

            rgb_images = images_in(base / "sRGB_images")
            if not save_middle(rgb_images, out / f"{slug}_rgb.jpg"):
                missing.append(f"{slug} sRGB")

            ir_images = images_in(find_ir_folder(base))
            if not ir_images and fruit == "Banana" and slug_label == "spoiled":
                # In TR-6 the spoiled banana's thermal images sit under Normal/Banana;
                # keep only those taken on the spoiled-labelled days (filenames start YYYYMMDD).
                spoiled_dates = {p.name[:8] for p in rgb_images}
                ir_images = [p for p in images_in(find_ir_folder(root / "Normal" / "Banana"))
                             if p.name[:8] in spoiled_dates]
            if not save_middle(ir_images, out / f"{slug}_ir.jpg"):
                missing.append(f"{slug} thermal")

    if missing:
        print("\nNo images found for:", ", ".join(missing))
    else:
        print("\nAll 24 sample images exported.")


if __name__ == "__main__":
    main()
