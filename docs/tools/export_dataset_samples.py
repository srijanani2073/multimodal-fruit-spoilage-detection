"""
Export one matched sRGB + thermal pair for every TR-6 class
(6 produce types x {not spoiled, spoiled}) into the website's assets/dataset/ folder.

Run from the repository root on the machine that has the dataset:

    python docs/tools/export_dataset_samples.py --root ~/Desktop/Thesis/TR-6

Every class is sampled at the same relative point, so the twelve pairs are comparable:

  * day      the day at --position along the class's recording days (default 0.5, the middle);
             only days with both an sRGB and a thermal image in the chosen session count
  * session  --session (default afternoon); if that day has no matching pair in it, the next
             session in the order afternoon, morning, evening is used and reported
  * images   the first sRGB image of that session, and the thermal image taken closest in time
             to it, so both cameras show the specimen at the same moment

The chosen date, session, both file names and the time gap between them are printed and saved to
assets/dataset/samples.json. Existing images in assets/dataset/ are overwritten.
"""
import argparse
import json
import re
from datetime import datetime
from pathlib import Path

from PIL import Image

FRUITS = ["Banana", "Carrot", "Guava", "Indian_Gooseberry", "Mango", "Tomato"]
LABELS = {"Not_spoiled": "notspoiled", "Spoiled": "spoiled"}
IR_FOLDER_NAMES = ["IR_fusion_images", "IR_Fusion_images", "ir_fusion_images"]
SESSIONS = {"morning": (5, 12), "afternoon": (12, 16), "evening": (16, 19)}
SRGB_RE = re.compile(r"^(\d{8})_(\d{6})", re.IGNORECASE)
IR_RE = re.compile(r"(\d{8})_(\d{6})_[\d.]+C_[\d.]+C\.(?:jpe?g|png)$", re.IGNORECASE)
MAX_SIDE = 900


def find_ir_folder(base: Path):
    for name in IR_FOLDER_NAMES:
        if (base / name).exists():
            return base / name
    return None


def session_of(hhmmss: str):
    hour = int(hhmmss[:2])
    for name, (lo, hi) in SESSIONS.items():
        if lo <= hour < hi:
            return name
    return None


def index(folder, pattern):
    """{(date, session): [(datetime, path), ...] sorted by time}"""
    out = {}
    if folder is None or not folder.exists():
        return out
    for p in folder.iterdir():
        m = pattern.search(p.name)
        if not m:
            continue
        session = session_of(m.group(2))
        if session:
            t = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
            out.setdefault((m.group(1), session), []).append((t, p))
    for v in out.values():
        v.sort()
    return out


def pick_pair(rgb, ir, position, session_order):
    """Choose (date, session, rgb_path, ir_path, gap_seconds) or None."""
    paired = {key for key in rgb if key in ir}
    if not paired:
        return None
    # Count days along the trajectory using the first session (in preference order) that is
    # present on at least one day, so every class is sampled in the same session where possible.
    for session in session_order:
        days = sorted({d for d, s in paired if s == session})
        if days:
            break
    day = days[round(position * (len(days) - 1))]
    for session in session_order:
        if (day, session) in paired:
            t_rgb, p_rgb = rgb[(day, session)][0]
            t_ir, p_ir = min(ir[(day, session)], key=lambda x: abs((x[0] - t_rgb).total_seconds()))
            return day, session, p_rgb, p_ir, abs((t_ir - t_rgb).total_seconds())
    return None


def save(src: Path, out_path: Path):
    img = Image.open(src).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    img.save(out_path, quality=86, optimize=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="~/Desktop/Thesis/TR-6", help="TR-6 dataset root")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "assets" / "dataset"))
    parser.add_argument("--position", type=float, default=0.5,
                        help="relative point along each class's days: 0 = first day, 1 = last (default 0.5)")
    parser.add_argument("--session", default="afternoon", choices=list(SESSIONS))
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    order = [args.session] + [s for s in ["afternoon", "morning", "evening"] if s != args.session]
    record, missing = {}, []

    for fruit in FRUITS:
        for folder_label, slug_label in LABELS.items():
            base = root / "Classified" / fruit / folder_label
            slug = f"{fruit.lower()}_{slug_label}"
            rgb = index(base / "sRGB_images", SRGB_RE)
            ir = index(find_ir_folder(base), IR_RE)
            if fruit == "Banana" and slug_label == "spoiled":
                # TR-6 stores most of the spoiled banana's thermal images under Normal/Banana:
                # fill every (date, session) of the spoiled sRGB days that has no thermal image.
                for key, shots in index(find_ir_folder(root / "Normal" / "Banana"), IR_RE).items():
                    if key in rgb and key not in ir:
                        ir[key] = shots

            pair = pick_pair(rgb, ir, args.position, order)
            if pair is None:
                missing.append(slug)
                print(f"{slug:<30} no day with both an sRGB and a thermal image "
                      f"(sRGB on {len({d for d, _ in rgb})} days, thermal on {len({d for d, _ in ir})} days; "
                      f"sRGB dates {min(rgb)[0] if rgb else '-'}..{max(rgb)[0] if rgb else '-'}, "
                      f"thermal dates {min(ir)[0] if ir else '-'}..{max(ir)[0] if ir else '-'})")
                continue
            day, session, p_rgb, p_ir, gap = pair
            save(p_rgb, out / f"{slug}_rgb.jpg")
            save(p_ir, out / f"{slug}_ir.jpg")
            note = "" if session == args.session else f"  ({args.session} missing that day)"
            print(f"{slug:<30} {day} {session:<9} {p_rgb.name} + {p_ir.name}  gap {gap / 60:.1f} min{note}")
            record[slug] = {"date": day, "session": session, "rgb": p_rgb.name, "ir": p_ir.name,
                            "gap_minutes": round(gap / 60, 1)}

    with open(out / "samples.json", "w") as f:
        json.dump({"position": args.position, "session": args.session, "samples": record}, f, indent=2)
    if missing:
        print("\nNo matched pair for:", ", ".join(missing))
    else:
        print(f"\nAll 12 matched pairs exported to {out}")


if __name__ == "__main__":
    main()
