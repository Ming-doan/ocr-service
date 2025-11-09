import os
import re
import json
import requests
from pathlib import Path
from urllib.parse import urlparse
from zipfile import ZipFile

# Configuration
PDF_FOLDER = "pdfs"  # folder containing pdfs
OUTPUT_FOLDER = "output"
LOG_FILE = "logs.json"
API_URL = "http://localhost:8001/api/pdf/extract"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

logs = []

def log_entry(pdf_name, status, msg=""):
    logs.append({"pdf": pdf_name, "status": status, "message": msg})

def fetch_pdf_files(folder):
    return [f for f in Path(folder).glob("*.pdf")]

def send_api_request(pdf_path):
    with open(pdf_path, "rb") as f:
        files = {"pdf": f}
        data = {
            "response_format": "merge",
            "merge_algorithm": "table_aware",
            "max_pages": 0
        }
        try:
            response = requests.post(API_URL, files=files, data=data, timeout=None)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise RuntimeError(f"API request failed: {e}")

def save_markdown(pdf_name, text):
    md_file = Path(OUTPUT_FOLDER) / (pdf_name.stem + ".md")
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(text)
    return md_file

def fetch_images(md_path):
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    image_urls = re.findall(r"!\[.*?\]\((https?:\/\/[^\)]+)\)", content)
    replacements = {}

    for idx, url in enumerate(image_urls, start=1):
        ext = os.path.splitext(urlparse(url).path)[1] or ".png"
        image_name = f"image_{idx}{ext}"
        image_path = md_path.parent / image_name
        try:
            resp = requests.get(url)
            resp.raise_for_status()
            with open(image_path, "wb") as img_f:
                img_f.write(resp.content)
            replacements[url] = f"|<image_{idx}>|"
        except Exception as e:
            log_entry(md_path.stem, "image_error", f"{url}: {e}")

    # Replace images in markdown
    for old, new in replacements.items():
        content = content.replace(old, new)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(content)

def zip_files(pdf_name, folder):
    zip_path = Path(folder) / (pdf_name.stem + ".zip")
    with ZipFile(zip_path, "w") as zipf:
        for file in Path(OUTPUT_FOLDER).glob(f"{pdf_name.stem}.*"):
            zipf.write(file, arcname=file.name)

def main():
    pdf_files = fetch_pdf_files(PDF_FOLDER)

    for pdf_path in pdf_files:
        try:
            log_entry(pdf_path.name, "start")
            resp_json = send_api_request(pdf_path)
            result_text = resp_json.get("data", {}).get("result", "").replace("\n", "")
            md_file = save_markdown(pdf_path, result_text)
            fetch_images(md_file)
            zip_files(pdf_path, OUTPUT_FOLDER)
            log_entry(pdf_path.name, "success")
        except Exception as e:
            log_entry(pdf_path.name, "error", str(e))

    # Save logs
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)

if __name__ == "__main__":
    main()
