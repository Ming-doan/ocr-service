import os
from io import BytesIO
from datetime import timedelta
from urllib.parse import urlparse

from minio import Minio

from core.base import BaseService


class MinioService(Minio, BaseService):
    def __init__(self):
        self.minio_download_url = os.getenv("MINIO_DOWNLOAD_URL", "http://localhost:9000")

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
    
    def _get_url_path(self, url: str) -> str:
        u = urlparse(url)
        result = u.path or "/"
        if u.query:
            result += f"?{u.query}"
        if u.fragment:
            result += f"#{u.fragment}"
        return result
    
    def get_access_url(
        self,
        bucket_name: str,
        object_name: str,
        expires: int = 3600,
    ) -> str:
        url = self.presigned_get_object(
            bucket_name=bucket_name,
            object_name=object_name,
            expires=timedelta(seconds=expires),
        )
        return f"{self.minio_download_url}{self._get_url_path(url)}"