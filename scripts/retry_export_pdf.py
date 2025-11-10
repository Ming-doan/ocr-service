import os
from pathlib import Path
from subprocess import run
import argparse

PDF_FOLDER = "pdfs"
OUTPUT_FOLDER = "output"
MAIN_SCRIPT = "scripts/export_pdf_public_test.py"


def output_is_empty(stem: str) -> bool:
    folder = Path(OUTPUT_FOLDER) / stem
    return (not folder.exists()) or (not any(folder.iterdir()))


def find_empty_outputs():
    pdf_files = Path(PDF_FOLDER).glob("*.pdf")
    failed = []

    for pdf in pdf_files:
        stem = pdf.stem
        if output_is_empty(stem):
            failed.append(stem)

    return sorted(failed)


def rerun(stems):
    for stem in stems:
        pdf_path = Path(PDF_FOLDER) / f"{stem}.pdf"
        if not pdf_path.exists():
            print(f"[skip] missing PDF: {pdf_path}")
            continue

        print(f"[retry] {pdf_path}")
        run(["python3", MAIN_SCRIPT, "--single", str(pdf_path)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, help="Retry only this PDF stem")
    args = parser.parse_args()

    if args.only:
        rerun([args.only])
        return

    failed = find_empty_outputs()

    if not failed:
        print("✅ No empty output folders. Nothing to retry.")
        return

    print("❗ Empty output detected:", failed)
    rerun(failed)


if __name__ == "__main__":
    main()
