"""Project startup entrypoint."""
from __future__ import annotations

import argparse
import os
import socket
import sys
from pathlib import Path

import uvicorn
from sqlalchemy import text

ROOT_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = ROOT_DIR / "研发"

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app.core.config import settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Legal RAG System Server")
    parser.add_argument(
        "--host",
        type=str,
        default=settings.HOST,
        help=f"Bind host (default: {settings.HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.PORT,
        help=f"Bind port (default: {settings.PORT})",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=settings.DEBUG,
        help="Enable auto reload in development mode",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=settings.WORKERS if not settings.DEBUG else 1,
        help=f"Worker count (default: {settings.WORKERS})",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=settings.LOG_LEVEL.lower(),
        choices=["debug", "info", "warning", "error", "critical"],
        help="Log level",
    )

    args = parser.parse_args()

    print_banner(args)

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level=args.log_level,
        access_log=True,
    )


def print_banner(args: argparse.Namespace) -> None:
    print("\n" + "=" * 72)
    print("Legal RAG System")
    print(f"Version : {settings.VERSION}")
    print(f"Host    : {args.host}")
    print(f"Port    : {args.port}")
    print(f"Reload  : {'on' if args.reload else 'off'}")
    print(f"Workers : {args.workers if not args.reload else 1}")
    print("=" * 72)

    _print_access_urls(args.host, args.port)

    print("\nService checks:")
    check_services()


def _print_access_urls(host: str, port: int) -> None:
    bind_all = host in {"0.0.0.0", "::"}
    local_host = "127.0.0.1" if bind_all else host

    print("\nAccess URLs:")
    print(f"  Web     : http://{local_host}:{port}/")
    print(f"  API     : http://{local_host}:{port}{settings.API_V1_STR}")
    print(f"  Docs    : http://{local_host}:{port}/docs")
    print(f"  Health  : http://{local_host}:{port}/health")

    if bind_all:
        lan_ips, primary_ip = _get_lan_ipv4_addresses()
        if lan_ips:
            print("\nLAN share URLs:")
            if primary_ip:
                print(f"  Recommended : http://{primary_ip}:{port}/")
            for ip in lan_ips:
                if ip != primary_ip:
                    print(f"  Alternate   : http://{ip}:{port}/")
        else:
            print("\nLAN share URLs:")
            print("  No LAN IPv4 address detected.")
    else:
        print("\nLAN share URLs:")
        print("  Disabled for the current host binding.")
        print("  Start with --host 0.0.0.0 to let others on the same network open it.")


def _get_lan_ipv4_addresses() -> tuple[list[str], str | None]:
    addresses: set[str] = set()
    primary_ip: str | None = None

    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, family=socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127."):
                addresses.add(ip)
    except OSError:
        pass

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("192.0.2.1", 80))
            ip = sock.getsockname()[0]
            if ip and not ip.startswith("127."):
                addresses.add(ip)
                primary_ip = ip
    except OSError:
        pass

    sorted_addresses = sorted(addresses)
    if not primary_ip and sorted_addresses:
        primary_ip = sorted_addresses[0]

    return sorted_addresses, primary_ip


def check_services() -> None:
    try:
        from app.models.database import engine

        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("  MySQL  : connected")
    except Exception as exc:
        print(f"  MySQL  : failed - {str(exc)[:80]}")

    print("  Redis  : skipped")
    print("  Milvus : skipped")


if __name__ == "__main__":
    main()
