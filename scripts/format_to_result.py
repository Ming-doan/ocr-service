import os
import re
import zipfile
import shutil
from pathlib import Path
from bs4 import BeautifulSoup
from minio import Minio
from minio.error import S3Error
import argparse

RESULT_DIR = "tmp"
OUTPUT_ZIP = "OCR.zip"

############################################################
# CONFIG
############################################################

MINIO_CLIENT = Minio(
    "42.96.34.158:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False,
)
MINIO_BUCKET = "ocr-images"

CUSTOM_REMOVE_STRINGS = [
    "VIETTEL A1 RACE",
    "CÁC THUẬT NGỮ CƠ BẢN TRÊN WEBSITE BÁN HÀNG",
    "TD427",
    "MÔ TẢ VÙNG CẠNH MÁY TÍNH HP",
    "*",
]

############################################################
# REGEX DEFINITIONS
############################################################

RE_REMOVE_H1 = re.compile(r"^# .*$", flags=re.MULTILINE)
RE_REMOVE_A = re.compile(r"^#\s*A\d?$", flags=re.MULTILINE)
RE_REMOVE_VIETTEL = re.compile(r"(?i)VIETTEL AI RACE")
RE_REMOVE_LAN_BAN_HANH = re.compile(r"(?i)Lần ban hành\s*:?\s*\d+")
RE_FIX_NUMBERED_TITLE = re.compile(r"^(#+)\s+\d+(\.\d+)*\s+", flags=re.MULTILINE)
RE_REMOVE_TD = re.compile(r"\bT\.?\s*D\.?\s*\d*\b", flags=re.IGNORECASE)

RE_MARKDOWN_IMAGE = re.compile(r"!\[.*?\]\((.*?)\)", flags=re.IGNORECASE)
RE_NORMALIZE_IMAGE_PLACEHOLDER = re.compile(
    r"\|\s*<?\s*image[_\s-]?(\d+)\s*>?\s*\|", flags=re.IGNORECASE
)

############################################################
# HELPERS
############################################################


def remove_custom_strings(md_text: str) -> str:
    for s in CUSTOM_REMOVE_STRINGS:
        p = re.compile(rf"[*#\s]*{re.escape(s)}[*\s]*", flags=re.IGNORECASE)
        md_text = p.sub("", md_text)
    return md_text


def normalize_image_placeholders(md_text: str) -> str:
    return RE_NORMALIZE_IMAGE_PLACEHOLDER.sub(
        lambda m: f"|<image_{m.group(1)}>|", md_text
    )


def clean_image_name(img_name: str) -> str:
    """
    Remove the trailing UUID/hash before file extension.
    Example:
        Public_..._12_2_398641010c.png → Public_..._12_2.png
    """
    return re.sub(r"_[0-9a-fA-F]{6,}\.(png|jpg|jpeg)$", r".\1", img_name)


def is_zero_zero_image(img_name: str) -> bool:
    """
    Detect pattern *_0_0_* before the trailing hash or extension.
    Example:
        Public_..._0_0_abc.png → True
        Public_..._12_2_abc.png → False
    """
    return bool(re.search(r"_0_0(_|[0-9a-fA-F])", img_name))


def fetch_minio_image(img_name: str, save_path: Path):
    """Download image from MinIO bucket to local save_path."""
    clean_name = clean_image_name(img_name)
    try:
        MINIO_CLIENT.fget_object(MINIO_BUCKET, clean_name, str(save_path))
        return True
    except S3Error as e:
        print(f"⚠ Failed to fetch {img_name} → {clean_name}: {e}")
        return False
    except Exception as e:
        print(f"⚠ Error fetching {img_name}: {e}")
        return False


def replace_and_fetch_images(md_text: str, img_folder: Path) -> str:
    """
    Find markdown images, fetch them from MinIO, and replace with |<image_X>|.
    Ignore images with *_0_0_* pattern.
    """
    matches = list(RE_MARKDOWN_IMAGE.finditer(md_text))
    counter = 1

    for m in matches:
        img_path = m.group(1)
        img_name = os.path.basename(img_path.strip())
        if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        # Skip zero_zero images
        if is_zero_zero_image(img_name):
            md_text = md_text.replace(m.group(0), "")
            continue

        local_img = img_folder / img_name
        if not local_img.exists():
            fetch_minio_image(img_name, local_img)

        placeholder = f"|<image_{counter}>|"
        md_text = md_text.replace(m.group(0), placeholder)
        counter += 1

    return md_text


############################################################
# TABLE MERGING
############################################################


def merge_adjacent_tables(md_text: str) -> str:
    soup = BeautifulSoup(md_text, "html.parser")
    tables = soup.find_all("table")
    i = 0

    while i < len(tables) - 1:
        t1 = tables[i]
        t2 = tables[i + 1]

        between = []
        node = t1.next_sibling
        while node and node != t2:
            between.append(node)
            node = node.next_sibling

        is_ok = True
        image_placeholders = []
        for b in between:
            text = str(b).strip()
            if not text:
                continue
            if re.fullmatch(r"\|\<image_\d+\>\|", text):
                image_placeholders.append(text)
                continue
            is_ok = False
            break

        if not is_ok:
            i += 1
            continue

        for tbl in (t1, t2):
            if not tbl.find("tbody"):
                tb = soup.new_tag("tbody")
                for row in tbl.find_all("tr"):
                    tb.append(row)
                tbl.append(tb)

        t1_body = t1.find("tbody")
        t2_body = t2.find("tbody")

        r1 = t1.find("tr")
        r2 = t2.find("tr")
        if not r1 or not r2:
            i += 1
            continue

        cols1 = len(r1.find_all(["td", "th"]))
        cols2 = len(r2.find_all(["td", "th"]))

        if cols1 == cols2:
            for row in t2_body.find_all("tr"):
                t1_body.append(row)

            t2.decompose()
            tables.pop(i + 1)

            after = t1
            for ph in image_placeholders:
                p = soup.new_tag("p")
                p.string = ph
                after.insert_after(p)
                after = p
        else:
            i += 1

    return str(soup)


############################################################
# ZIP PROCESSING
############################################################


def process_zip(zip_path: Path, answer_list: list[str]):
    name = zip_path.stem
    out_folder = Path(RESULT_DIR) / name
    img_folder = out_folder / "images"
    img_folder.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        md_file = None
        for file in z.namelist():
            if file.lower().endswith(".md"):
                md_file = file
        if not md_file:
            print(f"⚠ No markdown in {zip_path.name}")
            return
        md_text = z.read(md_file).decode("utf-8")

    ############################################################
    # PROCESSING STEPS
    ############################################################

    md_text = replace_and_fetch_images(md_text, img_folder)
    md_text = normalize_image_placeholders(md_text)
    md_text = remove_custom_strings(md_text)
    md_text = RE_REMOVE_H1.sub("", md_text)
    md_text = RE_REMOVE_A.sub("", md_text)
    md_text = RE_REMOVE_VIETTEL.sub("", md_text)
    md_text = RE_REMOVE_LAN_BAN_HANH.sub("", md_text)
    md_text = RE_FIX_NUMBERED_TITLE.sub(r"\1 ", md_text)
    md_text = RE_REMOVE_TD.sub("", md_text)
    # md_text = merge_adjacent_tables(md_text)

    (out_folder / "main.md").write_text(md_text, encoding="utf-8")
    answer_list.append(f"# {name}\n\n{md_text}\n\n")


############################################################
# MAIN
############################################################


def main():
    Path(RESULT_DIR).mkdir(exist_ok=True)
    answer_list = []

    parser = argparse.ArgumentParser()
    parser.add_argument("--res-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--single", type=str, help="Process a single PDF file path")
    args = parser.parse_args()

    for zip_path in Path(f"./{args.res_dir}").glob("Public_*.zip"):
        print(f"Processing {zip_path.name}...")
        process_zip(zip_path, answer_list)

    Path(RESULT_DIR, "answer.md").write_text(
        "\n\n".join(answer_list), encoding="utf-8"
    )

    if Path(args.out_dir).exists():
        Path(args.out_dir).unlink()

    shutil.make_archive("OCR", "zip", RESULT_DIR)
    print("✅ Completed → OCR.zip created")


if __name__ == "__main__":
    main()
