from typing import cast
from PIL import Image
import re
from pydantic import BaseModel

import json

from shapely.geometry import Polygon
from shapely.strtree import STRtree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from bs4 import BeautifulSoup

from core.base import BaseService
from services.ocr.service import OCRService, ExtractionResult



def save_json_debug(data, filename: str):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


class TableAwareMergeConfig(BaseModel):
    removal_regex_patterns: list[str] = [
        r"^A$|^A.$",
        r'(?i)\bVIETTEL AI RACE\b',
        r'(?i)Lần ban hành\s*:?\s*\d+'
    ]  # Patterns to remove results before processing
    bbox_overlap_threshold: float = 0.5  # > X% overlap to consider overlapping
    diff_word_freq_threshold: float = 0.1  # < X% difference to consider similar
    diff_word_num_component_threshold: float = 0.8  # >= X% of component difference to consider similar
    start_page_offset: int = 0  # Skip first N pages for header analysis
    end_page_offset: int = 0  # Skip last N pages for footer analysis
    begin_num_result_content: int = 2  # Compare first N results to determine general header layout
    end_num_result_content: int = 0  # Compare last N results to determine general footer layout


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


_ListOfPageCombinedResults = list[_PageCombinedResults | _MergedPageCombinedResults]


class TableAwareMergeService(BaseService):
    def __init__(self):
        self.ocr_service = OCRService.provider()

        self.image_bbox_scale_factor = (1.0, 1.0)

    def _prune_results_by_regex(
        self,
        results: list[ExtractionResult],
        patterns: list[str]
    ) -> list[ExtractionResult]:
        """Remove ExtractionResult items whose text matches any regex pattern"""
        compiled_patterns = [re.compile(p) for p in patterns]
        pruned_results = []
        for r in results:
            text = r.text.strip()
            if not any(p.match(text) for p in compiled_patterns):
                pruned_results.append(r)
        return pruned_results
    
    def _remove_components_by_category(
        self,
        results: list[ExtractionResult],
        categories: list[str]
    ) -> list[ExtractionResult]:
        return [r for r in results if r.category not in categories]

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
                (r.bbox[0], r.bbox[3]),
            ])
            for r in results
        ]

        index = STRtree(polys)

        # Map geometry object → index via STRtree.geometries
        geom_to_idx = {geom: i for i, geom in enumerate(index.geometries)}

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

                # query neighbors
                candidates = index.query(polys[idx])

                for cand in candidates:
                    j = geom_to_idx.get(cand)
                    if j is None or j in visited:
                        continue

                    if self._iou(polys[idx], polys[j]) >= threshold:
                        stack.append(j)

            # merge cluster
            cluster_items = [results[k] for k in cluster]
            groups.append(
                _CombinedResults(
                    results=cluster_items,
                    text="\n".join(r.text for r in cluster_items).strip()
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
        texts = []
        for r in results:
            if r.category == "Picture":
                texts.append(" ")
            else:
                texts.append(r.text.strip())
        if len(texts) <= 1:
            return 0.0

        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(lowercase=True)
        X = vectorizer.fit_transform(texts)  # shape: (N, V) → sparse matrix

        # centroid of all vectors, still sparse, shape (1, V)
        centroid = X.mean(axis=0)  # type: ignore

        # convert to regular array (dense)
        centroid = np.asarray(centroid)

        # cosine similarity for each page vs centroid
        sims = cosine_similarity(X, centroid)

        # convert similarity → difference (0–1)
        diffs = 1 - sims

        # return average difference
        return float(np.mean(diffs))

    def _is_component_index_general(
        self,
        group: list[_CombinedResults],
        config: TableAwareMergeConfig,
    ) -> bool:
        if len(group) <= 1:
            return False

        flat = self._flat_combined_results(group)
        if not flat:
            return False

        # Compute per-page diff
        diffs = []
        for r in group:
            r_flat = self._flat_combined_results([r])
            if not r_flat:
                continue
            diff = self._get_result_diff_by_word_feq(r_flat)
            diffs.append(diff)

        if not diffs:
            return False

        # Count how many pages have "similar" component
        similar_count = sum(d <= config.diff_word_freq_threshold for d in diffs)
        ratio = similar_count / len(diffs)

        return ratio >= config.diff_word_num_component_threshold

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
            group = [p.results[idx] for p in pages if idx < len(p.results)]
            if len(group) < 2:
                continue

            if self._is_component_index_general(group, config):
                header.append(
                    _PageCombinedResults(
                        page_number=first_page.page_number,
                        results=[first_page.results[idx]],
                    )
                )
                used_header_indices.add(idx)

        # 3. FOOTER DETECTION
        max_footer_idx = config.end_num_result_content

        for rev_idx in range(1, max_footer_idx + 1):
            group = [p.results[-rev_idx] for p in pages if rev_idx <= len(p.results)]
            if len(group) < 2:
                continue

            if self._is_component_index_general(group, config):
                footer.append(
                    _PageCombinedResults(
                        page_number=last_page.page_number,
                        results=[last_page.results[-rev_idx]],
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

    def _get_last_table_component_from_combined(
        self,
        c: list[_CombinedResults]
    ) -> _CombinedResults:
        for comp in c:
            for er in comp.results[::-1]:  # check from last to first
                if er.category == "Table":
                    return comp
        return c[-1]

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
                table_combined_blocks: list[_CombinedResults] = [self._get_last_table_component_from_combined(page.results)]

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
        flat: list[ExtractionResult] = []
        for combined in page.results:
            flat.extend(combined.results)

        return self.ocr_service.convert_to_markdown(
            image,
            flat,
            filename,
            image_bbox_scale_factor=self.image_bbox_scale_factor,
        )

    def _combined_results_to_text(
        self,
        images: list[Image.Image],
        filename: str,
        results: _ListOfPageCombinedResults
    ) -> str:
        chunks: list[str] = []

        for item in results:

            # -----------------------------------------
            # Case 1 — Normal page
            # -----------------------------------------
            if isinstance(item, _PageCombinedResults):
                chunks.append(self._render_page_to_markdown(
                    images[item.page_number - 1],
                    filename,
                    item
                ))
                continue

            # -----------------------------------------
            # Case 2 — Merged table block
            # -----------------------------------------
            if isinstance(item, _MergedPageCombinedResults):

                # 1) Render all *non-table* components from each page
                for page in item.results:
                    non_table = [
                        c for c in page.results
                        if not any(er.category == "Table" for er in c.results)
                    ]

                    if not non_table:
                        continue

                    # Build temp page with only non-table components
                    temp_page = _PageCombinedResults(
                        page_number=page.page_number,
                        results=non_table
                    )
                    chunks.append(self._render_page_to_markdown(
                        images[temp_page.page_number - 1],
                        filename,
                        temp_page
                    ))

                # 2) Append merged TABLE as final markdown content
                # item.text contains merged <table>…</table>
                if item.text.strip():
                    chunks.append(item.text)

                continue

        return "\n".join(chunks)

    def merge(
        self,
        images: list[Image.Image],
        filename: str,
        results: list[TableAwareResultInput],
        config: TableAwareMergeConfig,
    ) -> str:
        # 0. Prune results by regex
        pruned_results: list[TableAwareResultInput] = []
        for r in results:
            pruned_ocr_results = self._prune_results_by_regex(
                r.ocr_results,
                config.removal_regex_patterns
            )
            pruned_ocr_results = self._remove_components_by_category(
                pruned_ocr_results,
                categories=["Page-footer"]
            )
            pruned_results.append(
                TableAwareResultInput(
                    page_number=r.page_number,
                    ocr_results=pruned_ocr_results
                )
            )
        # save_json_debug(
        #     [r.model_dump() for r in pruned_results],
        #     "tmp/step0.json"
        # )

        # 1. Get overlapping OCR results
        page_combined_results: list[_PageCombinedResults] = []
        for result in pruned_results:
            page_combined_results.append(
                _PageCombinedResults(
                    page_number=result.page_number,
                    results=self._get_overlap_ocr_results(result.ocr_results, config)
                )
            )
        # save_json_debug(
        #     [p.model_dump() for p in page_combined_results],
        #     "tmp/step1.json"
        # )

        # 2. Get general layout
        layout_results = self._get_general_layout_from_pages_results(
            page_combined_results,
            config
        )
        # save_json_debug(
        #     layout_results.model_dump(),
        #     "tmp/step2.json"
        # )

        # 3. Merge table results in each section
        merged_contents = self._merge_table_from_results(
            layout_results.contents
        )
        # save_json_debug(
        #     [p.model_dump() for p in merged_contents],
        #     "tmp/step3.json"
        # )

        # 4. Get markdown text of all parts
        page_cover = self._combined_results_to_text(
            images,
            filename,
            cast(_ListOfPageCombinedResults, layout_results.begin_page_cover)
        )
        header = self._combined_results_to_text(
            images,
            filename,
            cast(_ListOfPageCombinedResults, layout_results.header)
        )
        contents = self._combined_results_to_text(
            images,
            filename,
            merged_contents
        )
        footer = self._combined_results_to_text(
            images,
            filename,
            cast(_ListOfPageCombinedResults, layout_results.footer)
        )
        end_page_cover = self._combined_results_to_text(
            images,
            filename,
            cast(_ListOfPageCombinedResults, layout_results.end_page_cover)
        )

        return "\n\n".join([
            page_cover,
            header,
            contents,
            footer,
            end_page_cover,
        ])