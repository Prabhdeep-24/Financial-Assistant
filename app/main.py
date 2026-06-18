"""FastAPI entrypoint for the AI Financial Document Analyst."""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm the embedding model and vector store so the first upload
    # doesn't trigger a cold ONNX model download mid-request.
    from app.llm import get_embeddings
    from app.state import get_or_build_vectorstore
    get_embeddings()
    get_or_build_vectorstore()
    yield


app = FastAPI(
    lifespan=lifespan,
    title="AI Financial Document Analyst",
    description=(
        "Analyzes 10-K/10-Q filings and earnings call transcripts: financial "
        "metric extraction, period-over-period comparisons, management tone "
        "analysis, risk factor tracking, competitor benchmarking, and "
        "investment memo generation."
    ),
    version="0.1.0",
)

app.include_router(router, prefix="/api")

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def index() -> FileResponse:
    """Serve the dashboard frontend."""
    return FileResponse(STATIC_DIR / "index.html")


@app.exception_handler(RuntimeError)
def llm_failure_handler(request: Request, exc: RuntimeError) -> JSONResponse:
    """LLM-backed endpoints raise RuntimeError when the provider is unavailable
    (e.g. free-tier rate limits exhausted). Surface this as 503, not a 500 traceback."""
    return JSONResponse(
        status_code=503,
        content={"detail": f"LLM provider unavailable, please retry later: {exc}"},
    )