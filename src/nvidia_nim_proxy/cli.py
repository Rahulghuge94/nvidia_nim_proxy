"""Command-line interface for the NVIDIA NIM proxy."""

from __future__ import annotations

import argparse
import os

from .server import DEFAULT_MODEL, NIM_DEFAULT_BASE, ProxyConfig, run_server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NVIDIA NIM compatibility proxy")
    parser.add_argument("--host", default="127.0.0.1", help="Proxy listen host")
    parser.add_argument("--port", type=int, default=8000, help="Proxy listen port")
    parser.add_argument(
        "--upstream",
        default=os.getenv("NVIDIA_NIM_UPSTREAM", NIM_DEFAULT_BASE),
        help="NVIDIA NIM upstream URL",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("NVIDIA_API_KEY") or os.getenv("NVDIA_API_KEY"),
        help="NVIDIA API key. Prefer NVIDIA_API_KEY; NVDIA_API_KEY is accepted for compatibility.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("NVIDIA_NIM_MODEL", DEFAULT_MODEL),
        help="Default model used when a client does not send one.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.getenv("NVIDIA_NIM_TIMEOUT", "300")),
        help="Upstream timeout in seconds.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> ProxyConfig:
    return ProxyConfig(
        host=args.host,
        port=args.port,
        upstream_base=args.upstream,
        api_key=args.api_key,
        default_model=args.model,
        timeout_seconds=args.timeout,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_server(config_from_args(args))
