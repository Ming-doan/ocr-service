from PIL import Image
from pydantic import BaseModel

from shapely.geometry import Polygon
from shapely.strtree import STRtree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from bs4 import BeautifulSoup

from core.base import BaseService
from services.ocr.service import OCRService, ExtractionResult


class TableAwareMergeConfig(BaseModel):
    bbox_overlap_threshold: float = 0.5  # > X% overlap to consider overlapping
    diff_word_freq_threshold: float = 0.05  # < X% difference to consider similar
    start_page_offset: int = 0  # Skip first N pages for header analysis
    end_page_offset: int = 0  # Skip last N pages for footer analysis
    begin_num_result_content: int = 5  # Compare first N results to determine general header layout
    end_num_result_content: int = 5  # Compare last N results to determine general footer layout
    removal_regex_patterns: list[str] = []  # Patterns to remove from final text (e.g., page numbers)


class TableAwareResultInput(BaseModel):
    page_number: int
    ocr_results: list[ExtractionResult]


class _CombinedResults(BaseModel):
    results: list[ExtractionResult]
    text: str = ""


class _PageCombinedResults(BaseModel):
    page_number: int
    results: list[_CombinedResults]


class _LayoutCombinedResults(BaseModel):
    begin_page_cover: list[_PageCombinedResults]
    header: list[_PageCombinedResults]
    contents: list[_PageCombinedResults]
    footer: list[_PageCombinedResults]
    end_page_cover: list[_PageCombinedResults]


class _MergedPageCombinedResults(BaseModel):
    results: list[_PageCombinedResults]
    text: str = ""


class TableAwareMergeService(BaseService):
    def __init__(self):
        self.ocr_service = OCRService.provider()

    def _iou(self, p1: Polygon, p2: Polygon) -> float:
        inter = p1.intersection(p2).area
        if inter == 0:
            return 0.0
        union = p1.area + p2.area - inter
        return inter / union

    def _get_overlap_ocr_results(
        self,
        results: list[ExtractionResult],
        config: TableAwareMergeConfig
    ) -> list[_CombinedResults]:
        if not results:
            return []

        threshold = config.bbox_overlap_threshold

        # convert bbox → shapely polygon
        polys = [
            Polygon([
                (r.bbox[0], r.bbox[1]),
                (r.bbox[2], r.bbox[1]),
                (r.bbox[2], r.bbox[3]),
                (r.bbox[0], r.bbox[3])
            ])
            for r in results
        ]

        index = STRtree(polys)
        visited = set()
        groups = []

        for i in range(len(polys)):
            if i in visited:
                continue

            stack = [i]
            cluster = []

            while stack:
                idx = stack.pop()
                if idx in visited:
                    continue

                visited.add(idx)
                cluster.append(idx)

                # spatial index query
                candidates = index.query(polys[idx])

                for cand_poly in candidates:
                    j = polys.index(cand_poly)
                    if j in visited:
                        continue

                    iou = self._iou(polys[idx], polys[j])

                    if iou >= threshold:
                        stack.append(j)

            # Build merged result for this cluster
            cluster_items = [results[k] for k in cluster]
            merged_text = "\n".join(r.text for r in cluster_items).strip()

            groups.append(
                _CombinedResults(
                    results=cluster_items,
                    text=merged_text
                )
            )

        return groups

    def _flat_combined_results(
        self,
        results: list[_CombinedResults]
    ) -> list[ExtractionResult]:
        flat: list[ExtractionResult] = []
        for cr in results:
            # Dummy extraction result for compare text
            flat.append(ExtractionResult(
                bbox=[0, 0, 0, 0],
                category="Text",
                text="\n".join(r.text for r in cr.results).strip(),
            ))
        return flat

    def _get_result_diff_by_word_feq(
        self,
        results: list[ExtractionResult]
    ) -> float:
        texts = [r.text.strip() for r in results if r.text.strip()]
        if len(texts) <= 1:
            return 0.0
        
        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(lowercase=True)
        X = vectorizer.fit_transform(texts)  # shape: (N, V)

        # centroid of all vectors
        centroid = X.mean(axis=0)  # type: ignore[attr-defined]

        # cosine similarity for each page vs centroid
        sims = cosine_similarity(X, centroid)

        # convert similarity → difference (0–1)
        diffs = 1 - sims

        # return average difference
        return float(np.mean(diffs))

    def _get_general_layout_from_pages_results(
        self,
        results: list[_PageCombinedResults],
        config: TableAwareMergeConfig,
    ) -> _LayoutCombinedResults:
        # 0. Select pages for analysis
        total_pages = len(results)
        start = config.start_page_offset
        end = total_pages - config.end_page_offset
        pages = results[start:end]

        # Prepare containers (page containers, not raw components)
        begin_page_cover: list[_PageCombinedResults] = []
        header: list[_PageCombinedResults] = []
        contents: list[_PageCombinedResults] = []
        footer: list[_PageCombinedResults] = []
        end_page_cover: list[_PageCombinedResults] = []

        # 1. BEGIN / END PAGE COVERS
        if config.start_page_offset > 0:
            begin_page_cover = results[:config.start_page_offset]

        if config.end_page_offset > 0:
            end_page_cover = results[-config.end_page_offset:]

        # Handle trivial case
        if len(pages) <= 1:
            return _LayoutCombinedResults(
                begin_page_cover=begin_page_cover,
                header=header,
                contents=pages,
                footer=footer,
                end_page_cover=end_page_cover,
            )

        # Define page layout containers
        first_page = pages[0]
        last_page = pages[-1]
        used_header_indices = set()
        used_footer_indices = set()

        # 2. HEADER DETECTION
        max_header_idx = config.begin_num_result_content

        for idx in range(max_header_idx):
            # gather that component index across pages
            group = []
            for p in pages:
                if idx < len(p.results):
                    group.append(p.results[idx])
            if len(group) < 2:
                continue

            # diff based on ExtractionResults
            flat = self._flat_combined_results(group)
            if not flat:
                continue

            diff = self._get_result_diff_by_word_feq(flat)
            if diff <= config.diff_word_freq_threshold:
                # ✅ detected header → keep only first page's component
                header.append(
                    _PageCombinedResults(
                        page_number=first_page.page_number,
                        results=[first_page.results[idx]]
                    )
                )
                used_header_indices.add(idx)

        # 3. FOOTER DETECTION
        max_footer_idx = config.end_num_result_content

        for rev_idx in range(1, max_footer_idx + 1):
            group = []
            for p in pages:
                if rev_idx <= len(p.results):
                    group.append(p.results[-rev_idx])
            if len(group) < 2:
                continue

            flat = self._flat_combined_results(group)
            if not flat:
                continue

            diff = self._get_result_diff_by_word_feq(flat)
            if diff <= config.diff_word_freq_threshold:
                # ✅ detected footer → keep only last page's component
                footer.append(
                    _PageCombinedResults(
                        page_number=last_page.page_number,
                        results=[last_page.results[-rev_idx]]
                    )
                )
                used_footer_indices.add(len(last_page.results) - rev_idx)

        # 4. CONTENT REMAINDER
        for p in pages:
            new_results = []
            for i, comp in enumerate(p.results):
                if i in used_header_indices:
                    continue
                if i in used_footer_indices:
                    continue
                new_results.append(comp)

            contents.append(
                _PageCombinedResults(
                    page_number=p.page_number,
                    results=new_results
                )
            )

        return _LayoutCombinedResults(
            begin_page_cover=begin_page_cover,
            header=header,
            contents=contents,
            footer=footer,
            end_page_cover=end_page_cover,
        )
    
    def _has_header(self, html: str) -> bool:
        soup = BeautifulSoup(html, "html.parser")
        return soup.find("th") is not None

    def _get_column_count(self, html: str) -> int:
        soup = BeautifulSoup(html, "html.parser")
        tr = soup.find("tr")
        if not tr:
            return 0
        return len(tr.find_all(["td", "th"]))

    def _merge_table_html(self, html_list: list[str]) -> str:
        """Merge multiple table HTMLs by concatenating rows."""
        soups = [BeautifulSoup(h, "html.parser") for h in html_list]
        base = soups[0]
        base_tbody = base.find("tbody") or base.find("table")
        if not base_tbody:
            return html_list[0]

        for s in soups[1:]:
            tb = s.find("tbody") or s.find("table")
            if tb:
                for tr in tb.find_all("tr"):
                    base_tbody.append(tr)

        return str(base)

    def _combined_is_table(self, c: _CombinedResults) -> bool:
        if not c.results:
            return False
        for er in c.results[::-1]:  # check from last to first
            if er.category == "Table":
                return True
        return False

    def _merge_table_from_results(
        self,
        results: list[_PageCombinedResults]
    ) -> list[_PageCombinedResults | _MergedPageCombinedResults]:
        merged: list[_PageCombinedResults | _MergedPageCombinedResults] = []
        i = 0
        n = len(results)

        while i < n:
            page = results[i]

            # If this page's last combined result is a table candidate and next page exists and
            # its first combined result is also a table candidate -> start a chain
            if page.results and \
                self._combined_is_table(page.results[-1]) and \
                (i + 1) < n and \
                results[i + 1].results and \
                self._combined_is_table(results[i + 1].results[0]):
                
                # --- produce pre-table part from current page (everything before last combined) ---
                if len(page.results) > 1:
                    pre_page = _PageCombinedResults(
                        page_number=page.page_number,
                        results=page.results[:-1]
                    )
                    merged.append(pre_page)
                # else: nothing before table on this page

                # --- collect chain pages and table combined blocks ---
                chain_pages: list[_PageCombinedResults] = [page]  # will be used to record page_numbers in merged.results
                table_combined_blocks: list[_CombinedResults] = [page.results[-1]]  # last combined of first page

                j = i + 1
                # extend chain while next page's first combined is a table
                while j < n and \
                    results[j].results and \
                    self._combined_is_table(results[j].results[0]):

                    chain_pages.append(results[j])
                    table_combined_blocks.append(results[j].results[0])
                    j += 1

                # --- create merged HTML from the table combined blocks' .text ---
                # Each combined block's .text should be the table html fragment.
                html_merged = self._merge_table_html([c.text for c in table_combined_blocks])

                # For merged.results we keep page stubs preserving page_number and order.
                # We put empty results because the table content is now in text.
                merged_page_stubs: list[_PageCombinedResults] = [
                    _PageCombinedResults(page_number=p.page_number, results=[]) for p in chain_pages
                ]

                merged.append(
                    _MergedPageCombinedResults(
                        results=merged_page_stubs,
                        text=html_merged
                    )
                )

                # --- handle post-table content of the last chain page (if any) ---
                last_chain_page = chain_pages[-1]
                # if last_chain_page had more components after the first (table) combined,
                # keep them as a new page entry appearing after the merged block.
                if len(last_chain_page.results) > 1:
                    post_page = _PageCombinedResults(
                        page_number=last_chain_page.page_number,
                        results=last_chain_page.results[1:]
                    )
                    merged.append(post_page)

                # advance i to j (we consumed pages up to j-1)
                i = j
                continue

            # otherwise, no chain starting at this page -> keep page unchanged
            merged.append(page)
            i += 1

        return merged

    def _render_page_to_markdown(
        self,
        image: Image.Image,
        filename: str,
        page: _PageCombinedResults
    ) -> str:
        # flatten list[_CombinedResults] → list[ExtractionResult]
        flat: list[ExtractionResult] = []
        for combined in page.results:
            flat.extend(combined.results)

        return self.ocr_service.convert_to_markdown(
            image, flat, filename
        )

    def _combined_results_to_text(
        self,
        images: list[Image.Image],
        filename: str,
        results: list[_PageCombinedResults | _MergedPageCombinedResults]
    ) -> str:
        chunks: list[str] = []

        for item in results:

            # Case 1: item is a merged group (contains many pages)
            if isinstance(item, _MergedPageCombinedResults):
                for page in item.results:
                    # Filter not Table pages
                    if all(r.category != "Table" for r in page.results):
                        continue
                    page_md = self._render_page_to_markdown(
                        images[page.page_number - 1], filename, page
                    )
                    chunks.append(page_md)

            # Case 2: item is a single page
            else:  # _PageCombinedResults
                page_md = self._render_page_to_markdown(
                    images[page.page_number - 1], filename, item
                )
                chunks.append(page_md)

        return "\n".join(chunks)

    def merge(
        self,
        images: list[Image.Image],
        filename: str,
        results: list[TableAwareResultInput],
        config: TableAwareMergeConfig,
    ) -> str:
        # 1. Get overlapping OCR results
        page_combined_results: list[_PageCombinedResults] = []
        for result in results:
            page_combined_results.append(
                _PageCombinedResults(
                    page_number=result.page_number,
                    results=self._get_overlap_ocr_results(result.ocr_results, config)
                )
        )
        
        # 2. Get general layout
        layout_results = self._get_general_layout_from_pages_results(
            page_combined_results,
            config
        )

        # 3. Merge table results in each section
        merged_contents = self._merge_table_from_results(
            layout_results.contents
        )

        return ""