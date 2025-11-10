from typing import Awaitable
import fitz  # PyMuPDF
from PIL import Image
import asyncio
import uuid
from io import BytesIO

import json

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
            pdf = fitz.open(stream=data, filetype="pdf")
        except Exception as e:
            raise ValueError("Failed to open PDF document") from e

        images = []

        try:
            for page_idx in range(pdf.page_count):
                if max_pages and page_idx >= max_pages:
                    break

                page = pdf.load_page(page_idx)

                # ✅ Render directly using DPI (replaces Matrix(dpi/72))
                pix = page.get_pixmap(dpi=self.pdf_dpi)

                # ✅ Safeguard: very rare case when image becomes too large (>4500px)
                if pix.width > 4500 or pix.height > 4500:
                    # Render at default 72 DPI instead
                    pix = page.get_pixmap(dpi=72)

                mode = "RGBA" if pix.alpha else "RGB"
                img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)

                # Always provide clean RGB
                img = img.convert("RGB")

                images.append(img)

            return images

        finally:
            pdf.close()

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