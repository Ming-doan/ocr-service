import os
from typing import Callable, TypeVar, TypedDict
from functools import partial
import asyncio

T = TypeVar("T")

MAX_CONCURRENT_TASKS = int(os.getenv("MAX_CONCURRENT_TASKS", 1))  # Default to 1 to avoid out-of-memory in vllm
semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

async def run_in_async(func: Callable[..., T], *args, **kwargs) -> T:
    loop = asyncio.get_running_loop()
    # run_in_executor does NOT support kwargs → wrap with partial
    wrapped = partial(func, *args, **kwargs)
    async with semaphore:
        return await loop.run_in_executor(None, wrapped)
    

class HealthCheckFunc(TypedDict):
    name: str
    func: Callable[..., bool]
    

class ServiceStatus(TypedDict):
    service_name: str
    status: str
    error: str | None
    

def health_checker(
    functions: list[HealthCheckFunc]
) -> tuple[list[ServiceStatus], bool]:
    service_statuses: list[ServiceStatus] = []
    all_healthy = True

    for func in functions:
        service_name = func["name"]
        try:
            healthy = func["func"]()
            if healthy:
                service_statuses.append(ServiceStatus(
                    service_name=service_name,
                    status="healthy",
                    error=None,
                ))
            else:
                all_healthy = False
                service_statuses.append(ServiceStatus(
                    service_name=service_name,
                    status="unhealthy",
                    error="Health check returned False",
                ))
        except Exception as e:
            all_healthy = False
            service_statuses.append(ServiceStatus(
                service_name=service_name,
                status="unhealthy",
                error=str(e),
            ))

    return service_statuses, all_healthy