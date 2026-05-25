"""
compress_figures.py
Compress paper_latex/figures/ PNGs for Overleaf free plan compatibility.
Reduces pixel dimensions (300→150 DPI equivalent) + PNG optimization.
"""

from pathlib import Path
from PIL import Image
import sys

FIGURES_DIR = Path(__file__).parent.parent / "paper_latex" / "figures"

def compress_figure(path: Path, max_width_px: int = 1200) -> None:
    img = Image.open(path)
    orig_size = path.stat().st_size
    w, h = img.size

    # Resize if wider than max_width_px (halves typical 300-DPI figures)
    if w > max_width_px:
        ratio = max_width_px / w
        new_h = int(h * ratio)
        img = img.resize((max_width_px, new_h), Image.LANCZOS)

    # Save optimized PNG (lossless but smaller)
    img.save(path, "PNG", optimize=True, compress_level=9)
    new_size = path.stat().st_size
    pct = (1 - new_size / orig_size) * 100
    print(f"  {path.name:40s}  {orig_size//1024:5d} KB -> {new_size//1024:4d} KB  ({pct:+.0f}%)")


def main():
    print(f"\nCompressing figures in: {FIGURES_DIR}\n")
    total_before = 0
    total_after = 0

    for png in sorted(FIGURES_DIR.glob("fig*.png")):
        before = png.stat().st_size
        total_before += before
        compress_figure(png)
        total_after += png.stat().st_size

    print(f"\nTotal: {total_before//1024} KB -> {total_after//1024} KB "
          f"({(1-total_after/total_before)*100:+.0f}%)")


if __name__ == "__main__":
    main()
