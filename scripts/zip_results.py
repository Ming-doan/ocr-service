import os
import zipfile

def zip_all_zip_files(src_dir: str, output_zip: str):
    zip_files = [f for f in os.listdir(src_dir) if f.lower().endswith(".zip")]

    if not zip_files:
        print("No .zip files found.")
        return

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for f in zip_files:
            path = os.path.join(src_dir, f)
            z.write(path, arcname=f)

    print(f"Created: {output_zip}")

if __name__ == "__main__":
    # example
    zip_all_zip_files("output", "pdfs/all_zips.zip")
