import os
import json

from redis import Redis

from core.base import BaseService


class RedisService(Redis, BaseService):
    def __init__(self):
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", 6379))
        redis_db = int(os.getenv("REDIS_DB", 0))
        super().__init__(
            host=redis_host,
            port=redis_port,
            db=redis_db,
        )

    def set(
        self,
        key: str,
        value: dict,
        expire: int | None = None,
    ):
        super().set(key, json.dumps(value), ex=expire)

    def get(
        self,
        key: str,
    ) -> dict | None:
        value = super().get(key)
        if value is not None:
            return json.loads(value)  # type: ignore
        return None