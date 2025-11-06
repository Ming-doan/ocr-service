from typing import Annotated
from fastapi import APIRouter, UploadFile, File, Form

from core.interfaces.api_interface import ApiResponse
from core.interfaces.api_interface import OCRResponseFormat, DefaultOCRResponseFormat
from services.ocr.service import OCRService, ExtractionResult


router = APIRouter()


@router.post("/extract", response_model=ApiResponse[str | list[ExtractionResult]])
async def extract_from_image(
    image: Annotated[UploadFile, File(...)],
    response_format: Annotated[OCRResponseFormat, Form(...)] = DefaultOCRResponseFormat,
):
    service = OCRService.provider()

    image_data = await image.read()
    result = await service.extract(
        data=image_data,
        filename=image.filename,
        mode=response_format,
    )

    return ApiResponse(
        message="Extraction successful",
        data=result,
    )

    