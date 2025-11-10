from PIL import Image
from io import BytesIO
import re
import uuid

from core.base import BaseService
from core.utils import run_in_async
from core.interfaces.api_interface import OCRResponseFormat, DefaultOCRResponseFormat
from core.services.ocr_llm import OCRLLMService, ExtractionResult
from core.services.minio import MinioService


class OCRService(BaseService):
    def __init__(self):
        self.ocr_llm_service = OCRLLMService.provider()
        self.minio_service = MinioService.provider()

        self.image_bucket = "ocr-images"
        self.minio_service.create_bucket(self.image_bucket)
        self.image_expire_seconds = 604800  # 7 days
        self.header_levels = {
            "Page-header": 1,
            "Section-header": 2,
            "Title": 3,
        }
    
    def _normalize_header(self, text: str) -> str:
        return re.sub(r'^#+\s*', '', text).strip()
    
    def convert_to_markdown(
        self,
        image: Image.Image,
        results: list[ExtractionResult],
        filename: str | None = None,
        image_bbox_scale_factor: tuple[float, float] = (1.0, 1.0)
    ) -> str:
        markdown_parts = []
        filename = filename or str(uuid.uuid4())
        img_idx = 0

        for page_idx, page_result in enumerate(results):
            if page_result.category == 'List-item':
                markdown_parts.append(f"{page_result.text.strip()}\n\n")

            elif page_result.category == 'Caption':
                markdown_parts.append(f"*{page_result.text.strip()}*\n\n")

            elif page_result.category == 'Footnote':
                markdown_parts.append(f"_{page_result.text.strip()}_\n\n")

            elif page_result.category == 'Formula':
                text = re.sub(r'^\${1,2}|\${1,2}$', '', page_result.text.strip())
                markdown_parts.append(f"${text}$\n\n")

            elif page_result.category == 'Table':
                markdown_parts.append(f"{page_result.text.strip()}\n\n")

            elif page_result.category == 'Text':
                markdown_parts.append(f"{page_result.text.strip()}\n\n")

            elif page_result.category == 'Page-footer':
                # Ignore page footer for markdown
                continue

            elif page_result.category == 'Picture':
                # Crop image using bbox
                bbox = page_result.bbox  # x0, y0, x1, y1
                bbox = (
                    bbox[0] * image_bbox_scale_factor[0],
                    bbox[1] * image_bbox_scale_factor[1],
                    bbox[2] * image_bbox_scale_factor[0],
                    bbox[3] * image_bbox_scale_factor[1],
                )
                cropped_image = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))

                # Upload to Minio
                img_buffer = BytesIO()
                cropped_image.save(img_buffer, format="PNG")
                object_name = f"{filename}_{page_idx}_{img_idx}.png"
                self.minio_service.upload(
                    bucket_name=self.image_bucket,
                    object_name=object_name,
                    data=img_buffer.getvalue(),
                )

                # Get access URL
                access_url = self.minio_service.get_access_url(
                    bucket_name=self.image_bucket,
                    object_name=object_name,
                    expires=self.image_expire_seconds,
                )

                # Add to markdown
                markdown_parts.append(f"![{filename}]({access_url})\n\n")
                img_idx += 1

            else:
                # Handle headers
                level = self.header_levels.get(page_result.category)
                if level:
                    _t = self._normalize_header(page_result.text)
                    markdown_parts.append(f"{'#' * level} {_t}\n\n")

        return "".join(markdown_parts)
    
    async def extract(
        self,
        data: bytes,
        filename: str | None = None,
        mode: OCRResponseFormat = DefaultOCRResponseFormat,
    ):
        image = Image.open(BytesIO(data))
        results = await run_in_async(self.ocr_llm_service.extract_text, image)

        if mode == "json":
            return results
        
        elif mode == "markdown":
            return self.convert_to_markdown(image, results, filename)