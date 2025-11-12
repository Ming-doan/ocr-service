import os
import re
import json
import requests
from pathlib import Path
from urllib.parse import urlparse
from zipfile import ZipFile
import argparse

# Configuration
PDF_FOLDER = "pdfs/input"          # Folder that contains PDFs
OUTPUT_FOLDER = "output"     # Where result folders + zip files go
LOG_FILE = "logs.json"
API_URL = "http://localhost:8005/api/pdf/extract"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

logs = []

# --- Logging (realtime) ------------------------------------------------------

def write_logs():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)

def log_entry(pdf_name, status, msg=""):
    logs.append({"pdf": pdf_name, "status": status, "message": msg})
    write_logs()


# --- Helpers -----------------------------------------------------------------

def fetch_pdf_files(folder):
    return [f for f in Path(folder).glob("*.pdf")]

def send_api_request(pdf_path):
    with open(pdf_path, "rb") as f:
        files = {"pdf": f}
        data = {
            "response_format": "merged",
            "merge_algorithm": "simple",
            "max_pages": 0
        }
        response = requests.post(API_URL, files=files, data=data, timeout=None)
        response.raise_for_status()
        return response.json()

def save_markdown(pdf_folder, pdf_name, text):
    md_path = pdf_folder / (pdf_name.stem + ".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(text)
    return md_path

def fetch_images(md_path):
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Capture full markdown + URL separately
    # group(0) = full: ![alt](url)
    # group(1) = alt text
    # group(2) = url
    matches = re.findall(r"!\[(.*?)\]\((https?:\/\/[^\)]+)\)", content)

    replacements = {}

    for idx, (alt, url) in enumerate(matches, start=1):
        ext = os.path.splitext(urlparse(url).path)[1] or ".png"
        image_name = f"image_{idx}{ext}"
        image_path = md_path.parent / image_name
        placeholder = f"|<image_{idx}>|"

        try:
            resp = requests.get(url)
            resp.raise_for_status()

            with open(image_path, "wb") as img_f:
                img_f.write(resp.content)

            full_md = f"![{alt}]({url})"
            replacements[full_md] = placeholder

        except Exception as e:
            log_entry(md_path.stem, "image_error", f"{url}: {e}")

    # Replace the whole ![alt](url) block
    for old, new in replacements.items():
        content = content.replace(old, new)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(content)


def zip_folder(pdf_name, pdf_folder):
    zip_path = Path(OUTPUT_FOLDER) / (pdf_name.stem + ".zip")
    with ZipFile(zip_path, "w") as zipf:
        for root, _, files in os.walk(pdf_folder):
            for f in files:
                full = Path(root) / f
                arc = full.relative_to(pdf_folder)  # keep folder structure clean
                zipf.write(full, arcname=arc)


# --- Main --------------------------------------------------------------------

def main(
    in_dir: str,
    out_dir: str
):
    pdf_files = fetch_pdf_files(in_dir)

    for idx, pdf_path in enumerate(pdf_files):
        name = pdf_path.name
        log_entry(name, "start")

        try:
            print(f"Processing {name}...")
            # Create per-PDF folder: output/myfile/
            pdf_folder = Path(out_dir) / pdf_path.stem
            pdf_folder.mkdir(parents=True, exist_ok=True)

            # 1. API call
            print(f"  - Sending API request...")
            resp_json = send_api_request(pdf_path)

            # 2. Extract and clean text
            print(f"  - Extracting text...")
            result_text = resp_json.get("data", {}).get("result", "")

            # 3. Save markdown
            print(f"  - Saving markdown...")
            md_path = save_markdown(pdf_folder, pdf_path, result_text)

            # 4. Download images + replace markdown syntax
            print(f"  - Fetching images...")
            fetch_images(md_path)

            # 5. Zip only this folder
            print(f"  - Creating zip file...")
            zip_folder(pdf_path, pdf_folder)

            log_entry(name, "success")

        except Exception as e:
            log_entry(name, "error", str(e))

    write_logs()


def main_single(
    pdf_path,
    out_dir: str
):
    name = pdf_path.name
    log_entry(name, "start")

    try:
        pdf_folder = Path(out_dir) / pdf_path.stem
        pdf_folder.mkdir(parents=True, exist_ok=True)

        resp_json = send_api_request(pdf_path)
        result_text = resp_json.get("data", {}).get("result", "")

        md_path = save_markdown(pdf_folder, pdf_path, result_text)
        fetch_images(md_path)
        zip_folder(pdf_path, pdf_folder)

        log_entry(name, "success")
    except Exception as e:
        log_entry(name, "error", str(e))

    write_logs()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--single", type=str, help="Process a single PDF file path")
    args = parser.parse_args()

    if args.single:
        main_single(Path(args.single), out_dir=args.out_dir)
    else:
        main(
            in_dir=args.in_dir,
            out_dir=args.out_dir
        )
