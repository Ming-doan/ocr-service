from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.interfaces.api_interface import ApiResponse
from services.ocr.router import router as ocr_router
from services.pdf_extractor.router import router as pdf_extractor_router


app = FastAPI(
    title="Viettel AI OCR Service",
    docs_url="/",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ocr_router, prefix="/api/ocr", tags=["OCR"])
app.include_router(pdf_extractor_router, prefix="/api/pdf", tags=["PDF Extractor"])

# Basic health check endpoint
@app.get("/health", tags=["Utils"])
async def health_check():
    return ApiResponse(message="Service is healthy")


# Exception handlers, routers, and other endpoints would be added here
@app.exception_handler(Exception)
async def global_exception_handler(_, exc):
    return ApiResponse(message="An error occurred", data={
        "detail": str(exc)
    }).as_json_response(status_code=500)
