import os
import sys
import picologging as logging

from typing import Annotated
from fastapi import FastAPI, Depends
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastui import prebuilt_html, FastUI
from fastui.auth import fastapi_auth_exception_handling, AuthRedirect

from cc_server.logging import logging_config, setup_logging
from cc_server import landing_page
from cc_server import auth
from cc_server import root
from cc_server import scheduler


def create_app(auth_token: str):
    app = FastAPI()
    logger = logging.getLogger("cc_server")

    fastapi_auth_exception_handling(app)
    app.include_router(landing_page.get_router(logger), prefix="/api/landing_page")
    app.include_router(auth.get_router(auth_token, logger), prefix="/api/auth")
    app.include_router(scheduler.get_router(logger), prefix="/api/schedule")
    app.include_router(root.get_router(), prefix="/api")

    @app.get("/robots.txt", response_class=PlainTextResponse)
    async def robots_txt() -> str:
        return "User-agent: *\nAllow: /"

    @app.get("/favicon.ico", status_code=404, response_class=PlainTextResponse)
    async def favicon_ico() -> str:
        return "page not found"

    @app.get("/{path:path}", response_class=HTMLResponse)
    async def html_landing() -> HTMLResponse:
        """Simple HTML page which serves the React app, comes last as it matches all paths."""
        return HTMLResponse(prebuilt_html(title="FastUI Demo"))

    return app


if __name__ == "__main__":
    import uvicorn

    auth_token = os.getenv("AUTH_TOKEN", None)
    if auth_token is None:
        logging.critical({"message": "Please set environment variable AUTH_TOKEN"})
        sys.exit(1)

    setup_logging()
    uvicorn.run(
        create_app(auth_token),
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8080)),
        log_level="debug",
        log_config=logging_config(),
    )
