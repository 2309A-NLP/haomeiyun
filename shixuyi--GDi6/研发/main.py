from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .api.routes import router
from .core.config import settings

app = FastAPI(title=settings.app_name, version="0.1.0")
app.include_router(router)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root() -> HTMLResponse:
    index_path = Path("static/index.html")
    css_path = Path("static/css/style.css")
    js_path = Path("static/js/app.js")
    asset_version = str(
        max(
            int(index_path.stat().st_mtime) if index_path.exists() else 0,
            int(css_path.stat().st_mtime) if css_path.exists() else 0,
            int(js_path.stat().st_mtime) if js_path.exists() else 0,
        )
    )
    html = index_path.read_text(encoding="utf-8").replace("__ASSET_VERSION__", asset_version)
    return HTMLResponse(
        content=html,
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@app.get("/api")
def api_root() -> dict:
    return {
        "name": settings.app_name,
        "message": "Prospectus RAG system is running.",
        "docs": "/docs",
    }
