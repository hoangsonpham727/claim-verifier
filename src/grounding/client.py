"""Vendor seam — all Isaacus model calls route through here."""
import os
from functools import lru_cache

from dotenv import load_dotenv
from isaacus import Isaacus, AsyncIsaacus

load_dotenv()


def _api_key() -> str:
    key = os.environ.get("ISAACUS_API_KEY")
    if not key:
        raise RuntimeError("ISAACUS_API_KEY not set in environment / .env")
    return key


@lru_cache(maxsize=1)
def get_client() -> Isaacus:
    return Isaacus(api_key=_api_key())


@lru_cache(maxsize=1)
def get_async_client() -> AsyncIsaacus:
    return AsyncIsaacus(api_key=_api_key())
