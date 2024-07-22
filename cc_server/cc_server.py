
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastui import prebuilt_html
from fastui.auth import fastapi_auth_exception_handling

from cc_server.landing_page import router as landing_page_router

app = FastAPI()

# fastapi_auth_exception_handling(app)
app.include_router(landing_page_router, prefix='/api')


@app.get('/robots.txt', response_class=PlainTextResponse)
async def robots_txt() -> str:
    return 'User-agent: *\nAllow: /'


@app.get('/favicon.ico', status_code=404, response_class=PlainTextResponse)
async def favicon_ico() -> str:
    return 'page not found'

@app.get('/{path:path}', response_class=HTMLResponse)
async def html_landing() -> HTMLResponse:
    """Simple HTML page which serves the React app, comes last as it matches all paths."""
    return HTMLResponse(prebuilt_html(title='FastUI Demo'))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8080, log_level="debug")
