from typing import Awaitable
import fitz  # PyMuPDF
from PIL import Image
import asyncio
import uuid
from io import BytesIO

from pydantic import BaseModel

from core.base import BaseService
from core.interfaces.api_interface import (
    PDFExtractionMode,
    DefaultPDFExtractionMode,
    PDFMergeAlgorithm,
    DefaultPDFMergeAlgorithm,
)
from core.services.minio import MinioService
from services.ocr.service import OCRService, ExtractionResult
from services.pdf_extractor.merge_services.table_aware import (
    TableAwareMergeConfig,
    TableAwareResultInput,
    TableAwareMergeService
)
from services.pdf_extractor.merge_services.table_aware_2 import (
    TableAware2MergeService
)


class _Result(BaseModel):
    page_number: int
    ocr_result: str | list[ExtractionResult]


class PDFExtractionResult(BaseModel):
    total_pages: int
    file: str
    result: str | list[_Result]


class PDFExtractorService(BaseService):
    def __init__(self):
        self.ocr_service = OCRService.provider()
        self.minio_service = MinioService.provider()
        self.table_aware_merge_service = TableAwareMergeService.provider()
        self.table_aware_2_merge_service = TableAware2MergeService.provider()

        self.pdf_bucket = "pdf-files"
        self.minio_service.create_bucket(self.pdf_bucket)
        self.pdf_dpi = 400
        self.pdf_access_expire_seconds = 604800  # 7 days

    def _pdf_to_images(
        self,
        data: bytes,
        max_pages: int | None = None,
    ) -> list[Image.Image]:
        try:
            pdf_document = fitz.open(stream=data, filetype="pdf")
        except Exception as e:
            raise ValueError("Failed to open PDF document") from e
        
        total_pages = pdf_document.page_count

        zoom_matrix = fitz.Matrix(self.pdf_dpi / 72, self.pdf_dpi / 72)
        images: list[Image.Image] = []

        try:
            for page_num in range(total_pages):
                if max_pages and page_num >= max_pages:
                    break

                page = pdf_document.load_page(page_num)
                pix = page.get_pixmap(matrix=zoom_matrix)  # type: ignore[attr-defined]

                # Handle transparency correctly
                mode = "RGBA" if pix.alpha else "RGB"
                image = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
                image = image.convert("RGB")  # Ensure no alpha channel

                images.append(image)
            return images

        finally:
            pdf_document.close()

    async def extract(
        self,
        data: bytes,
        filename: str | None = None,
        mode: PDFExtractionMode = DefaultPDFExtractionMode,
        merge_algorithm: PDFMergeAlgorithm = DefaultPDFMergeAlgorithm,
        merge_config: dict | None = None,
        max_pages: int | None = None,
    ):
        # Convert PDF to images
        images = self._pdf_to_images(data, max_pages=max_pages)

        # Generate unique filename
        if filename is None:
            filename = str(uuid.uuid4()) + ".pdf"
        else:
            filename = filename.rsplit(".", 1)[0] + str(uuid.uuid4())[:8] + ".pdf"

        # Save PDF to Minio, and get access URL
        self.minio_service.upload(
            bucket_name=self.pdf_bucket,
            object_name=filename,
            data=data,
        )
        access_url = self.minio_service.get_access_url(
            bucket_name=self.pdf_bucket,
            object_name=filename,
            expires=self.pdf_access_expire_seconds,
        )

        # Determine OCR mode
        if mode == "json":
            ocr_mode = "json"
        elif mode == "markdown":
            ocr_mode = "markdown"
        else:  # merged
            if merge_algorithm == "table_aware":
                ocr_mode = "json"
            else:  # simple
                ocr_mode = "markdown"

        # Perform OCR parallely
        tasks: list[Awaitable[str | list[ExtractionResult]]] = []
        for i, image in enumerate(images):
            buf = BytesIO()
            image.save(buf, format="JPEG")
            buf.seek(0)

            tasks.append(self.ocr_service.extract(
                data=buf.getvalue(),
                filename=f"{filename}_page_{i}.jpg",
                mode=ocr_mode,
            ))
        ocr_results = await asyncio.gather(*tasks)

        if mode == "json" or mode == "markdown":
            results: list[_Result] = []
            for page_number, ocr_result in enumerate(ocr_results):
                results.append(_Result(
                    page_number=page_number + 1,
                    ocr_result=ocr_result,
                ))
            return PDFExtractionResult(
                total_pages=len(images),
                file=access_url,
                result=results,
            )
        
        # merged
        if merge_algorithm == "table_aware":
            table_aware_result_inputs: list[TableAwareResultInput] = [
                TableAwareResultInput(
                    page_number=page_number + 1,
                    ocr_results=ocr_result,
                )
                for page_number, ocr_result in enumerate(ocr_results)
                if isinstance(ocr_result, list)
            ]
            result = self.table_aware_merge_service.merge(
                images=images,
                filename=filename,
                results=table_aware_result_inputs,
                config=TableAwareMergeConfig.model_validate(merge_config or {}),
            )

        elif merge_algorithm == "table_aware_2":
            result = self.table_aware_2_merge_service.merge()

        else:  # merged_algorithm: simple
            result = "\n\n".join(
                result if isinstance(result, str) else ""
                for result in ocr_results
            )
        return PDFExtractionResult(
            total_pages=len(images),
            file=access_url,
            result=result,
        )