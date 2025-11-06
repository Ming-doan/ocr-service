import os
from typing import Callable, TypeVar
from functools import partial
import asyncio

T = TypeVar("T")

MAX_CONCURRENT_TASKS = int(os.getenv("MAX_CONCURRENT_TASKS", 2))  # Default to 2 to avoid out-of-memory in vllm
semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

async def run_in_async(func: Callable[..., T], *args, **kwargs) -> T:
    loop = asyncio.get_running_loop()
    # run_in_executor does NOT support kwargs → wrap with partial
    wrapped = partial(func, *args, **kwargs)
    async with semaphore:
        return await loop.run_in_executor(None, wrapped)