from typing import Any, Dict
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import  FileResponse

from detector import CurrencyDetector


app = FastAPI(
    title="Counterfeit Currency Identification Agent",
    version="1.0.0",
    description="Offline-first Rs 500 note authenticity screening API.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = CurrencyDetector()
BASE_DIR = Path(__file__).resolve().parent


@app.get("/")
async def dashboard():
    dashboard_path = BASE_DIR / "index.html"

    if not dashboard_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard file not found.")

    return FileResponse(dashboard_path)


@app.get("/api/v1/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/scan")
async def scan_currency(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload must be an image file.")

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        np_buffer = np.frombuffer(contents, dtype=np.uint8)
        image = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image. Use JPEG, PNG, or WebP.")

        result = detector.analyze_note(image)
        if not result.get("success", False):
            raise HTTPException(status_code=422, detail=result)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected scan failure: {exc}") from exc
    finally:
        await file.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=7860,
        reload=True
    )
