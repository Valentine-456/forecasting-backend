from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

FRONTEND_DIST = Path(__file__).resolve().parent / "public"

app = FastAPI(title="UAV Forecast Backend")

app.mount(
    "/",
    StaticFiles(directory=FRONTEND_DIST, html=True),
    name="frontend",
)

@app.get("/")
def read_index():
    index_file = FRONTEND_DIST / "index.html"
    return FileResponse(index_file)
