import json
import os
from pathlib import Path
from subprocess import run

PDF_FOLDER = "pdfs"
OUTPUT_FOLDER = "output"
LOG_FILE = "logs.json"

# Your main extraction script
MAIN_SCRIPT = "export_pdf_public_test.py"


def load_logs():
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def output_is_empty(pdf_stem):
    folder = Path(OUTPUT_FOLDER) / pdf_stem
    return (not folder.exists()) or (not any(folder.iterdir()))


def find_failed_pdfs(logs):
    failed = set()

    for entry in logs:
        name = entry["pdf"]
        stem = Path(name).stem

        if entry["status"] == "error":
            failed.add(stem)
            continue

        # success but empty output → treat as error
        if output_is_empty(stem):
            failed.add(stem)

    return sorted(failed)


def rerun(stems):
    for stem in stems:
        pdf_path = Path(PDF_FOLDER) / f"{stem}.pdf"
        if not pdf_path.exists():
            print(f"[skip] Missing PDF: {pdf_path}")
            continue

        print(f"[retry] {pdf_path}")
        run(["python3", MAIN_SCRIPT, "--single", str(pdf_path)])


def main():
    logs = load_logs()
    failed = find_failed_pdfs(logs)

    if not failed:
        print("✅ No failed files. Nothing to re-run.")
        return

    print("❗ Failed PDFs:", failed)
    rerun(failed)


if __name__ == "__main__":
    main()
