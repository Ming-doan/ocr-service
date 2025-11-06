import os
from io import BytesIO
from hashlib import sha256
from urllib.parse import quote

from minio import Minio

from core.base import BaseService

TEMP_DIR = "tmp/ocr_service"
os.makedirs(TEMP_DIR, exist_ok=True)


class MinioService(Minio, BaseService):
    def __init__(self):
        self.minio_download_url = os.getenv("MINIO_DOWNLOAD_URL", "http://localhost:8000")

        minio_endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
        minio_access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        minio_secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
        super().__init__(
            endpoint=minio_endpoint,
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            secure=False,
        )

    def create_bucket(
        self,
        bucket_name: str,
    ):
        if not self.bucket_exists(bucket_name):
            self.make_bucket(bucket_name)

    def upload(
        self,
        bucket_name: str,
        object_name: str,
        data: bytes,
    ):
        self.put_object(
            bucket_name=bucket_name,
            object_name=object_name,
            data=BytesIO(data),
            length=len(data),
        )

    def download(
        self,
        bucket_name: str,
        object_name: str,
    ) -> bytes:
        response = self.get_object(
            bucket_name=bucket_name,
            object_name=object_name,
        )
        data = response.read()
        response.close()
        response.release_conn()
        return data
    
    def get_access_url(
        self,
        bucket_name: str,
        object_name: str,
        expires: int = 3600,
    ) -> str:
        # download bytes
        data = self.download(bucket_name, object_name)

        # sanitize and build filename: keep original basename + short content hash + extension
        base = os.path.basename(object_name)
        name, ext = os.path.splitext(base)
        # simple sanitize: replace path separators and spaces
        safe_name = name.replace(os.path.sep, "_").replace(" ", "_")
        content_hash = sha256(data).hexdigest()[:10]
        filename = f"{safe_name}_{content_hash}{ext or ''}"

        # write file
        out_path = os.path.join(TEMP_DIR, filename)
        # atomic write: write to temp then rename
        tmp_path = out_path + ".tmp"
        with open(tmp_path, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, out_path)

        # return URL-encoded path suitable for use in a browser
        quoted = quote(filename)
        static_root = self.minio_download_url.rstrip("/")
        # assuming you mount tmp/ocr_service under /static
        return f"{static_root}/static/{quoted}"