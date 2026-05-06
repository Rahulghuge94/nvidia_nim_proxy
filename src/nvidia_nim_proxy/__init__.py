"""NVIDIA NIM compatibility proxy package."""

from .server import DEFAULT_MODEL, NIM_DEFAULT_BASE, ProxyConfig, run_server

__all__ = ["DEFAULT_MODEL", "NIM_DEFAULT_BASE", "ProxyConfig", "run_server"]

__version__ = "0.1.0"
