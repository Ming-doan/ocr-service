from abc import ABC
from functools import lru_cache


class BaseService(ABC):
    '''Base class for all services.'''
    @classmethod
    @lru_cache(maxsize=1)
    def provider(cls):
        return cls()