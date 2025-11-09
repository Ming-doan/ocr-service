from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from core.interfaces.api_interface import ApiResponse
from core.services.minio import MinioService
from core.services.ocr_llm import OCRLLMService
from core.utils import health_checker
from services.ocr.router import router as ocr_router
from services.pdf_extractor.router import router as pdf_extractor_router


app = FastAPI(
    title="OCR Service",
    docs_url="/",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="tmp/ocr_service"), name="static")

app.include_router(ocr_router, prefix="/api/ocr", tags=["OCR"])
app.include_router(pdf_extractor_router, prefix="/api/pdf", tags=["PDF Extractor"])

# Basic health check endpoint
@app.get("/health", tags=["Utils"])
def health_check():
    minio = MinioService.provider()
    ocr_llm = OCRLLMService.provider()

    health_status, is_service_ready = health_checker([
        {
            "name": f"MinIO Service - {minio.minio_endpoint}",
            "func": minio.health_check,
        },
        {
            "name": f"OCR LLM Service - {ocr_llm.ocr_endpoint}",
            "func": ocr_llm.health_check,
        },
    ])

    return ApiResponse(
        message="Health check completed",
        data=health_status,
    ).as_json_response(200 if is_service_ready else 503)


# Exception handlers, routers, and other endpoints would be added here
@app.exception_handler(Exception)
async def global_exception_handler(_, exc):
    return ApiResponse(message="An error occurred", data={
        "detail": str(exc)
    }).as_json_response(status_code=500)
