#!/usr/bin/env python
"""Diagnostic launcher for the Prospectus RAG system."""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

from app.core.logging import setup_logging


PROJECT_ROOT = Path(__file__).resolve().parent
LOG_FILE = PROJECT_ROOT / "logs" / "app.log"
CORE_DEPENDENCIES = ("fastapi", "uvicorn", "pydantic", "pydantic_settings")

logger = logging.getLogger("run")


def set_windows_utf8() -> None:
    if sys.platform != "win32":
        return

    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleCP(65001)
        kernel32.SetConsoleOutputCP(65001)
    except Exception:
        pass


def project_python_path() -> Path:
    if sys.platform == "win32":
        return PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    return PROJECT_ROOT / ".venv" / "bin" / "python"


def ensure_project_venv() -> None:
    target_python = project_python_path()
    if not target_python.exists():
        return

    current_python = Path(sys.executable).resolve()
    target_python = target_python.resolve()
    if current_python == target_python:
        return

    print(f"[INFO] Switching to project virtual environment: {target_python}")
    result = subprocess.run([str(target_python), *sys.argv], check=False, cwd=str(PROJECT_ROOT))
    raise SystemExit(result.returncode)


def check_python_version() -> None:
    if sys.version_info >= (3, 9):
        return

    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"[ERROR] Python 3.9+ is required. Current version: {version}")
    raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start the Prospectus RAG FastAPI service with diagnostics.")
    parser.add_argument("--host", help="Override APP_HOST from .env")
    parser.add_argument("--port", type=int, help="Override APP_PORT from .env")
    parser.add_argument("--reload", action="store_true", help="Force uvicorn reload on")
    parser.add_argument("--no-reload", action="store_true", help="Force uvicorn reload off")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the frontend automatically")
    parser.add_argument("--inspect-only", action="store_true", help="Run startup checks and print diagnostics only")
    parser.add_argument("--log-level", choices=("critical", "error", "warning", "info", "debug", "trace"))
    return parser.parse_args()


def resolve_log_level(level_name: str | None) -> int:
    levels = {
        "critical": logging.CRITICAL,
        "error": logging.ERROR,
        "warning": logging.WARNING,
        "info": logging.INFO,
        "debug": logging.DEBUG,
        "trace": logging.DEBUG,
    }
    return levels.get((level_name or "info").lower(), logging.INFO)


def configure_logging(level_name: str | None) -> None:
    level = resolve_log_level(level_name)
    setup_logging(level)
    logger.setLevel(level)


def validate_dependencies() -> None:
    missing: list[str] = []
    for module_name in CORE_DEPENDENCIES:
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)

    if missing:
        logger.error("Missing dependencies: %s", ", ".join(missing))
        logger.error("Install them with: %s -m pip install -r requirements.txt", sys.executable)
        raise SystemExit(1)

    logger.info("Dependency check passed: modules=%s", ", ".join(CORE_DEPENDENCIES))


def ensure_env_file() -> None:
    env_file = PROJECT_ROOT / ".env"
    env_example = PROJECT_ROOT / ".env.example"

    if env_file.exists():
        logger.info("Environment file ready: path=%s", env_file)
        return

    if env_example.exists():
        env_file.write_text(env_example.read_text(encoding="utf-8"), encoding="utf-8")
        logger.warning("Environment file was missing and has been created from template: path=%s", env_file)
        return

    logger.warning("Both .env and .env.example are missing: project_root=%s", PROJECT_ROOT)


def ensure_directories(settings) -> None:
    required_dirs = [
        settings.data_dir,
        settings.raw_pdf_path.parent,
        settings.processed_pdf_path.parent,
        settings.seed_qa_path.parent,
        settings.uploads_dir,
        settings.processed_docs_dir,
        settings.document_registry_path.parent,
        PROJECT_ROOT / "logs",
        PROJECT_ROOT / "tmp",
    ]
    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)
    logger.info("Directory check passed: ensured=%s", [str(path) for path in required_dirs])


def ensure_static_assets() -> None:
    index_file = PROJECT_ROOT / "static" / "index.html"
    if not index_file.exists():
        logger.error("Missing static entry file: path=%s", index_file)
        raise SystemExit(1)
    logger.info("Static asset check passed: entry=%s", index_file)


def resolve_reload(args: argparse.Namespace, app_debug: bool) -> bool:
    if args.reload and args.no_reload:
        logger.error("Use only one of --reload or --no-reload")
        raise SystemExit(1)
    if args.reload:
        return True
    if args.no_reload:
        return False
    return bool(app_debug)


def browser_host_for(host: str) -> str:
    return "127.0.0.1" if host in {"0.0.0.0", "::"} else host


def file_timestamp(path: Path) -> str | None:
    if not path.exists():
        return None
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime))


def json_record_count(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to inspect JSON records: path=%s error=%s", path, exc)
        return None
    return len(payload) if isinstance(payload, list) else None


def upload_file_count(uploads_dir: Path) -> int:
    if not uploads_dir.exists():
        return 0
    return sum(1 for path in uploads_dir.iterdir() if path.is_file())


def mask_secret(value: str | None) -> str:
    return "set" if value else "missing"


def log_path_status(label: str, path: Path, records: int | None = None) -> None:
    exists = path.exists()
    size_bytes = path.stat().st_size if exists and path.is_file() else None
    logger.info(
        "Asset status: name=%s exists=%s size_bytes=%s modified=%s records=%s path=%s",
        label,
        exists,
        size_bytes,
        file_timestamp(path),
        records,
        path,
    )


def log_runtime_summary(args: argparse.Namespace, settings) -> tuple[str, int, bool, str]:
    host = args.host or settings.app_host
    port = args.port or settings.app_port
    reload_enabled = resolve_reload(args, settings.app_debug)
    browser_host = browser_host_for(host)
    frontend_url = f"http://{browser_host}:{port}/"

    logger.info("Launcher started: cwd=%s project_root=%s python=%s", Path.cwd(), PROJECT_ROOT, sys.executable)
    logger.info(
        "Server summary: env=%s host=%s port=%s reload=%s debug=%s frontend=%s docs=%s log_file=%s",
        settings.app_env,
        host,
        port,
        reload_enabled,
        settings.app_debug,
        frontend_url,
        f"http://{browser_host}:{port}/docs",
        LOG_FILE,
    )
    logger.info(
        "Model summary: vector_backend=%s embedding_provider=%s rerank_provider=%s llm_provider=%s llm_model=%s llm_api_key=%s vlm_enabled=%s vlm_model=%s vlm_api_key=%s",
        settings.vector_backend,
        settings.embedding_provider,
        settings.rerank_provider,
        settings.llm_provider,
        settings.llm_model,
        mask_secret(settings.llm_api_key),
        settings.vlm_enabled,
        settings.vlm_model,
        mask_secret(settings.vlm_api_key),
    )
    logger.info(
        "Retrieval summary: vector_top_k=%s rerank_top_k=%s similarity_threshold=%s chunk_size=%s chunk_overlap=%s",
        settings.vector_top_k,
        settings.rerank_top_k,
        settings.similarity_threshold,
        settings.chunk_size,
        settings.chunk_overlap,
    )

    log_path_status("raw_pdf", settings.raw_pdf_path)
    log_path_status("processed_pdf", settings.processed_pdf_path)
    log_path_status("processed_text", settings.processed_text_path)
    log_path_status("document_registry", settings.document_registry_path, json_record_count(settings.document_registry_path))
    log_path_status("parsed_chunks", settings.parsed_chunks_path, json_record_count(settings.parsed_chunks_path))
    log_path_status("seed_qa", settings.seed_qa_path, json_record_count(settings.seed_qa_path))
    log_path_status("log_file", LOG_FILE)

    logger.info("Upload summary: upload_dir=%s file_count=%s", settings.uploads_dir, upload_file_count(settings.uploads_dir))

    from app.main import app

    route_paths = sorted(route.path for route in app.routes)
    logger.info("Application route summary: count=%s routes=%s", len(route_paths), route_paths)
    return host, port, reload_enabled, frontend_url


def warn_if_port_in_use(host: str, port: int) -> None:
    family = socket.AF_INET6 if ":" in host and host != "0.0.0.0" else socket.AF_INET
    bind_host = host
    if family == socket.AF_INET and host == "::":
        bind_host = "0.0.0.0"

    try:
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if family == socket.AF_INET6:
                sock.bind((bind_host, port, 0, 0))
            else:
                sock.bind((bind_host, port))
    except OSError as exc:
        logger.warning("Port probe indicates the port may already be in use: host=%s port=%s error=%s", host, port, exc)
        return

    logger.info("Port probe passed: host=%s port=%s", host, port)


def open_frontend_when_ready(url: str, delay_seconds: float = 1.0) -> None:
    def _open() -> None:
        time.sleep(delay_seconds)
        try:
            webbrowser.open(url)
            logger.info("Opened frontend in browser: url=%s", url)
        except Exception:
            logger.warning("Failed to open frontend automatically: url=%s", url, exc_info=True)

    threading.Thread(target=_open, daemon=True).start()


def start_server(args: argparse.Namespace, settings, host: str, port: int, reload_enabled: bool, frontend_url: str) -> None:
    import uvicorn

    log_level = (args.log_level or ("debug" if settings.app_debug else "info")).lower()
    warn_if_port_in_use(host, port)

    if not args.no_browser:
        open_frontend_when_ready(frontend_url)

    logger.info(
        "Starting uvicorn: app=%s host=%s port=%s reload=%s log_level=%s no_browser=%s",
        "app.main:app",
        host,
        port,
        reload_enabled,
        log_level,
        args.no_browser,
    )
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload_enabled,
        log_level=log_level,
    )


def main() -> None:
    os.chdir(PROJECT_ROOT)
    set_windows_utf8()
    ensure_project_venv()
    check_python_version()

    args = parse_args()
    configure_logging(args.log_level)

    try:
        validate_dependencies()
        ensure_env_file()

        from app.core.config import settings

        ensure_directories(settings)
        ensure_static_assets()
        host, port, reload_enabled, frontend_url = log_runtime_summary(args, settings)

        if args.inspect_only:
            logger.info("Inspect-only mode completed successfully")
            return

        start_server(args, settings, host, port, reload_enabled, frontend_url)
    except SystemExit:
        raise
    except Exception:
        logger.exception("Launcher failed")
        raise


if __name__ == "__main__":
    main()
