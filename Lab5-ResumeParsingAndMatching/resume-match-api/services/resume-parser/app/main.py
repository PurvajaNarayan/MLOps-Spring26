from pathlib import Path

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.parser import parse_resume
from app.models import ParsedResume

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(
    title="Resume Parser Service",
    version="0.1.0",
    description="Extracts structured data from resume PDFs",
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/parse", response_model=ParsedResume)
async def parse(resume: UploadFile):
    if not resume.filename or not resume.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    contents = await resume.read()
    if len(contents) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=400, detail="File too large (10MB max)")

    try:
        result = parse_resume(contents)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse PDF: {str(e)}")

    return result