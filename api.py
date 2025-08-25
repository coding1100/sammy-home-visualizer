# api.py
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse, Response
from tempfile import NamedTemporaryFile
from pathlib import Path
from typing import List, Optional

from analyze import analyze_image_to_memory  # from the earlier step

app = FastAPI(title="Sammy Vision API", version="0.1")

@app.post("/analyze")  # JSON response with the SVG string included
async def analyze_json(
    file: UploadFile = File(...),
    device: str = Query("cpu"),
    prompts: Optional[List[str]] = Query(None)
):
    suffix = Path(file.filename).suffix or ".jpg"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        svg_str, json_obj = analyze_image_to_memory(str(tmp_path), prompts=prompts, device=device)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    # <- SVG is included directly in the JSON under the "svg" key
    return JSONResponse({"svg": svg_str, "json": json_obj})


@app.post("/analyze.svg")  # Raw SVG response (handy for Postman / browsers)
async def analyze_svg(
    file: UploadFile = File(...),
    device: str = Query("cpu"),
    prompts: Optional[List[str]] = Query(None)
):
    suffix = Path(file.filename).suffix or ".jpg"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        svg_str, _ = analyze_image_to_memory(str(tmp_path), prompts=prompts, device=device)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    # <- Returns raw SVG so you can view it directly
    return Response(content=svg_str, media_type="image/svg+xml")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app",host="0.0.0.0",port=8000, reload=True)