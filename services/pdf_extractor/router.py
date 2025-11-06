from typing import Annotated
from fastapi import APIRouter, UploadFile, File, Form

from core.interfaces.api_interface import (
    ApiResponse,
    PDFExtractionMode,
    DefaultPDFExtractionMode,
    PDFMergeAlgorithm,
    DefaultPDFMergeAlgorithm,
)
from services.pdf_extractor.service import PDFExtractorService, PDFExtractionResult


router = APIRouter()


@router.post("/extract", response_model=ApiResponse[PDFExtractionResult])
async def extract_from_pdf(
    pdf: Annotated[UploadFile, File(...)],
    response_format: Annotated[PDFExtractionMode, Form(...)] = DefaultPDFExtractionMode,
    merge_algorithm: Annotated[PDFMergeAlgorithm, Form(...)] = DefaultPDFMergeAlgorithm,
    max_pages: Annotated[int, Form(...)] = 0,
):
    # Initialize service
    service = PDFExtractorService.provider()
    if max_pages <= 0:
        _max_pages = None
    else:
        _max_pages = max_pages

    # Read PDF data
    pdf_data = await pdf.read()

    # Call extraction service
    result = await service.extract(
        data=pdf_data,
        filename=pdf.filename,
        mode=response_format,
        merge_algorithm=merge_algorithm,
        max_pages=_max_pages,
    )

    return ApiResponse(
        message="PDF extraction successful",
        data=result,
    )