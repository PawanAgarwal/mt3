"""Convert a scanned piano score image into MusicXML or pitch lists.

This stand-alone script wraps the Audiveris Optical Music Recognition
engine to turn a photograph or scan of a single-page piano score into
MusicXML or a simple CSV / text representation of note events.

Usage examples
--------------
- Full MusicXML (written to ``out/score.musicxml``)::

    python image_to_musicxml.py --input IMG_3034.jpg --output out/score.musicxml

- Pitch list printed to stdout::

    python image_to_musicxml.py -i page.png --mode pitch

- Pitch list with durations written to ``out/pitches.csv``::

    python image_to_musicxml.py -i page1.pdf -m pitchdur -o out/pitches.csv
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


def _ensure_package(package: str, import_name: str | None = None) -> None:
    """Ensure that a required Python package is installed."""
    name = import_name or package
    try:
        importlib.import_module(name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])


# Attempt to import required packages, installing if necessary.
for pkg, name in [("music21", None), ("Pillow", "PIL"), ("opencv-python-headless", "cv2")]:
    _ensure_package(pkg, name)

# Imports after ensuring packages are present.
import cv2  # type: ignore
from music21 import converter  # type: ignore
from PIL import Image, ImageEnhance  # type: ignore


VERSION = "0.1"


def preprocess_image(
    path: Path, dpi: Optional[int], out_dir: Path
) -> Path:
    """Pre-process an image for OMR and return the temporary file path."""
    img = Image.open(path)
    if hasattr(img, "n_frames") and img.n_frames > 1:
        img.seek(0)

    if dpi is not None:
        scale = dpi / (img.info.get("dpi", (dpi, dpi))[0] or dpi)
        if scale != 1.0:
            new_size = (int(img.width * scale), int(img.height * scale))
            img = img.resize(new_size, Image.LANCZOS)
            img.info["dpi"] = (dpi, dpi)

    img = img.convert("L")
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_GRAY2BGR)
    blur = cv2.GaussianBlur(img_cv, (5, 5), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    coords = cv2.findNonZero(255 - thresh)
    angle = 0.0
    if coords is not None:
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
    center = (thresh.shape[1] // 2, thresh.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(thresh, M, (thresh.shape[1], thresh.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    pil_img = Image.fromarray(deskewed)
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(1.5)

    out_path = out_dir / "input.png"
    save_kwargs = {}
    if dpi is not None:
        save_kwargs["dpi"] = (dpi, dpi)
    pil_img.save(out_path, **save_kwargs)
    return out_path


def run_audiveris(img_path: Path, out_dir: Path) -> Path:
    """Run Audiveris on ``img_path`` and return path to MusicXML."""
    jar_path = Path(__file__).with_name("audiveris.jar")
    if not jar_path.exists():
        msg = (
            f"Audiveris JAR not found at {jar_path}. "
            "Please download it from https://audiveris.github.io/ and place "
            "it next to this script."
        )
        raise SystemExit(msg)
    if not shutil.which("java"):
        raise SystemExit(
            "Java executable not found. Please install Java and ensure it is in the PATH."
        )

    cmd = [
        "java",
        "-jar",
        str(jar_path),
        "-batch",
        "-export",
        "-output",
        str(out_dir),
        str(img_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(
            f"Audiveris failed: {result.stderr}\nCommand: {' '.join(cmd)}"
        )

    xml_files = list(out_dir.glob("*.musicxml"))
    if not xml_files:
        raise SystemExit("Audiveris did not produce a MusicXML file.")
    return xml_files[0]


def convert_image_to_musicxml(input_path: Path, dpi: Optional[int]) -> Path:
    """Convert an image or PDF to MusicXML using Audiveris."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dir = Path(tmpdir)
        processed = preprocess_image(input_path, dpi, tmp_dir)
        xml_path = run_audiveris(processed, tmp_dir)
        # Move XML out of temporary directory before it gets deleted.
        final_path = tmp_dir / "score.musicxml"
        shutil.move(xml_path, final_path)
        return final_path


def notes_to_pitch_list(notes: Iterable) -> str:
    return ", ".join(n.nameWithOctave for n in notes)


def notes_to_csv(notes: Iterable) -> str:
    lines = ["start_beat,pitch,duration_quarterLength"]
    for n in notes:
        lines.append(f"{n.offset},{n.nameWithOctave},{n.quarterLength}")
    return "\n".join(lines)


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Convert a scanned score to MusicXML or pitch lists.")
    parser.add_argument("--input", "-i", required=True, help="Path to image or PDF")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument(
        "--mode",
        "-m",
        choices=["musicxml", "pitch", "pitchdur"],
        default="musicxml",
        help="Output mode",
    )
    parser.add_argument("--dpi", type=int, help="Optional DPI to resample the image")
    parser.add_argument("--version", action="version", version=VERSION)

    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    xml_path = convert_image_to_musicxml(input_path, args.dpi)
    score = converter.parse(xml_path)

    if args.mode == "musicxml":
        dest = Path(args.output) if args.output else input_path.with_suffix(".musicxml")
        shutil.move(xml_path, dest)
        print(f"Wrote MusicXML to {dest}")
        return

    notes = list(score.recurse().notes)
    if args.mode == "pitch":
        text = notes_to_pitch_list(notes)
    else:
        text = notes_to_csv(notes)

    if args.output:
        Path(args.output).write_text(text)
    else:
        print(text)


if __name__ == "__main__":  # pragma: no cover
    main()
