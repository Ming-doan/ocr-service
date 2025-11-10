import os
import httpx
import json
import base64
import logging
from io import BytesIO
import re

from PIL import Image
from pydantic import BaseModel
from openai import OpenAI

from core.interfaces.api_interface import ExtractionCategory
from core.base import BaseService

logger = logging.getLogger("uvicorn.error")


PROMPT = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
"""

class ExtractionResult(BaseModel):
    bbox: list[int]
    category: ExtractionCategory
    text: str = ""


class OCRLLMService(OpenAI, BaseService):
    def __init__(self):
        self.ocr_model = os.environ.get("OCR_LLM_MODEL", "rednote-hilab/dots.ocr")
        self.max_completion_tokens = int(os.environ.get("OCR_LLM_MAX_TOKENS", 32768))

        self.ocr_endpoint = os.environ.get("OCR_LLM_ENDPOINT", "http://localhost:4377")
        super().__init__(
            api_key="0",
            base_url=f"{self.ocr_endpoint}/v1",
        )
        self.temperature = 0.1
        self.top_p = 0.9

    def health_check(self) -> bool:
        try:
            response = httpx.get(f"{self.ocr_endpoint}/health")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"OCR LLM health check failed: {e}")
            return False

    def image_to_base64(
        self,
        image: Image.Image,
        format: str = "PNG",
    ) -> str:
        buffered = BytesIO()
        image.save(buffered, format=format)
        base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/{format.lower()};base64,{base64_str}"
    
    def safe_json_loads(self, s: str):
        try:
            return json.loads(s)
        except:
            return []  # return empty list on failure

    def extract_text(
        self,
        image: Image.Image,
    ) -> list[ExtractionResult]:
        messages = []
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": self.image_to_base64(image)}
                },
                {
                    "type": "text",
                    "text": PROMPT,
                }
            ]
        })

        response = self.chat.completions.create(
            model=self.ocr_model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_completion_tokens
        )
        result_text = response.choices[0].message.content
        
        if not result_text:
            logger.error("OCR LLM returned empty result")
            return []
        
        result_json = self.safe_json_loads(result_text)
        extraction_results = [ExtractionResult.model_validate(item) for item in result_json]
        return extraction_results