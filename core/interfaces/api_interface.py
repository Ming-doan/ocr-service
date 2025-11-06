from typing import Generic, TypeVar, Optional, Literal
from pydantic import BaseModel
from fastapi.responses import JSONResponse

T = TypeVar('T')

# String Enums
OCRResponseFormat = Literal["markdown", "json"]
DefaultOCRResponseFormat = "markdown"

ExtractionCategory = Literal['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title']

PDFExtractionMode = Literal["markdown", "json", "merged"]
DefaultPDFExtractionMode = "merged"

PDFMergeAlgorithm = Literal["simple", "table_aware"]
DefaultPDFMergeAlgorithm = "table_aware"


# General Interfaces
class ApiResponse(BaseModel, Generic[T]):
    message: str
    data: Optional[T] = None

    def as_json_response(self, status_code: int = 200):
        return JSONResponse(
            status_code=status_code,
            content=self.model_dump(),
        )