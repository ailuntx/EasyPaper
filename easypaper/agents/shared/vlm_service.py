"""
Shared VLM Service
- **Description**:
    - Provides a unified VLM interface for all agents
    - Wraps VLMProvider/VLMFactory from vlm_review_agent
    - Adds high-level methods for figure/table analysis (used by Planner)
    - Used by both PlannerAgent and VLMReviewAgent
"""
import json
import logging
import io
from pathlib import Path
from typing import Optional, Dict, Any, List, TYPE_CHECKING
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..vlm_review_agent.providers.base import VLMProvider

logger = logging.getLogger("uvicorn.error")


# ---------------------------------------------------------------------------
# Analysis result models
# ---------------------------------------------------------------------------

class FigureAnalysis(BaseModel):
    """Result of VLM analysis on a figure image."""
    semantic_role: str = ""
    message: str = ""
    is_wide: bool = False
    suggested_section: str = ""
    caption_guidance: str = ""


class TableAnalysis(BaseModel):
    """Result of VLM analysis on a table image."""
    semantic_role: str = ""
    message: str = ""
    is_wide: bool = False
    suggested_section: str = ""


# ---------------------------------------------------------------------------
# Prompts for figure/table analysis
# ---------------------------------------------------------------------------

FIGURE_ANALYSIS_PROMPT = """Analyze this academic paper figure. Respond with JSON only.

Determine:
1. **semantic_role**: What role does this figure play? Choose one:
   architecture_overview, result_comparison, ablation_study, data_visualization,
   qualitative_example, pipeline_diagram, algorithm_illustration, general
2. **message**: In one sentence, what does this figure communicate to the reader?
3. **is_wide**: Should this figure span both columns in a two-column paper? (true/false)
   true if: complex diagram, comparison with many items, architecture overview, pipeline
   false if: simple chart, single plot, small illustration
4. **suggested_section**: Which section should this figure appear in?
   Choose: introduction, method, experiment, result, discussion
5. **caption_guidance**: Brief guidance on what the caption should emphasize.

Return ONLY this JSON:
{
    "semantic_role": "<role>",
    "message": "<one sentence>",
    "is_wide": <true/false>,
    "suggested_section": "<section>",
    "caption_guidance": "<guidance>"
}"""


TABLE_ANALYSIS_PROMPT = """Analyze this academic paper table image. Respond with JSON only.

Determine:
1. **semantic_role**: What role does this table play? Choose one:
   main_results, ablation, dataset_stats, comparison, hyperparameter, general
2. **message**: In one sentence, what does this table demonstrate?
3. **is_wide**: Should this table span both columns? (true if >5 columns or complex layout)
4. **suggested_section**: Which section should this table appear in?
   Choose: method, experiment, result, discussion

Return ONLY this JSON:
{
    "semantic_role": "<role>",
    "message": "<one sentence>",
    "is_wide": <true/false>,
    "suggested_section": "<section>"
}"""


# ---------------------------------------------------------------------------
# VLM Service
# ---------------------------------------------------------------------------

class VLMService:
    """
    Shared VLM service for all agents.
    - **Description**:
        - Creates a VLM provider from configuration
        - Provides high-level methods for figure/table analysis (Planner)
        - Provides low-level analyze_page for PDF review (VLMReviewAgent)
    """

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize VLM Service.

        - **Args**:
            - `provider` (str): Provider name (openai, claude, qwen)
            - `model` (str, optional): Model name (e.g. "gpt-4o")
            - `api_key` (str, optional): API key
            - `base_url` (str, optional): Custom base URL
        """
        self._provider_name = provider
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._provider: Optional["VLMProvider"] = None

    def _get_provider(self) -> "VLMProvider":
        """Lazy-initialize the VLM provider."""
        if self._provider is None:
            from ..vlm_review_agent.providers.base import VLMFactory
            kwargs: Dict[str, Any] = {}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._provider = VLMFactory.create(
                provider=self._provider_name,
                api_key=self._api_key or "",
                model=self._model or "gpt-4o",
                **kwargs,
            )
        return self._provider

    # ------------------------------------------------------------------
    # High-level: Figure / Table analysis (used by Planner)
    # ------------------------------------------------------------------

    async def analyze_figure(self, image_path: str) -> FigureAnalysis:
        """
        Analyze a figure image for planning purposes.

        - **Args**:
            - `image_path` (str): Path to figure image file

        - **Returns**:
            - `FigureAnalysis`: Semantic role, message, width, section suggestion
        """
        image_data = self._prepare_image_bytes_for_vlm(image_path)
        provider = self._get_provider()
        response = await provider.analyze_page(image_data, FIGURE_ANALYSIS_PROMPT)

        if not response.success:
            logger.warning("vlm_service.figure_analysis_failed: %s", response.error)
            return FigureAnalysis()

        parsed = self._parse_json(response.content or response.raw_response or "")
        return FigureAnalysis(
            semantic_role=parsed.get("semantic_role", ""),
            message=parsed.get("message", ""),
            is_wide=parsed.get("is_wide", False),
            suggested_section=parsed.get("suggested_section", ""),
            caption_guidance=parsed.get("caption_guidance", ""),
        )

    async def analyze_table_image(self, image_path: str) -> TableAnalysis:
        """
        Analyze a table image for planning purposes.

        - **Args**:
            - `image_path` (str): Path to table image file

        - **Returns**:
            - `TableAnalysis`: Semantic role, message, width, section suggestion
        """
        image_data = self._prepare_image_bytes_for_vlm(image_path)
        provider = self._get_provider()
        response = await provider.analyze_page(image_data, TABLE_ANALYSIS_PROMPT)

        if not response.success:
            logger.warning("vlm_service.table_analysis_failed: %s", response.error)
            return TableAnalysis()

        parsed = self._parse_json(response.content or response.raw_response or "")
        return TableAnalysis(
            semantic_role=parsed.get("semantic_role", ""),
            message=parsed.get("message", ""),
            is_wide=parsed.get("is_wide", False),
            suggested_section=parsed.get("suggested_section", ""),
        )

    # ------------------------------------------------------------------
    # Low-level: Page analysis (used by VLMReviewAgent)
    # ------------------------------------------------------------------

    async def analyze_page(self, image_data: bytes, prompt: str, **kwargs: Any) -> Any:
        """
        Analyze a PDF page image.
        - **Description**:
            - Delegates to the underlying VLM provider
            - Returns a VLMResponse object

        - **Args**:
            - `image_data` (bytes): PNG/JPEG image bytes
            - `prompt` (str): Analysis prompt
        """
        provider = self._get_provider()
        return await provider.analyze_page(image_data, prompt, **kwargs)

    async def analyze_pages(
        self, images: List[bytes], prompts: List[str], **kwargs: Any,
    ) -> list:
        """Analyze multiple pages. Delegates to provider."""
        provider = self._get_provider()
        return await provider.analyze_pages(images, prompts, **kwargs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_image_bytes_for_vlm(image_path: str) -> bytes:
        """
        Normalize visual input to PNG bytes for VLM providers.
        - **Description**:
            - If input is PDF, render first page to PNG.
            - If input is an image, re-encode to PNG to avoid provider-specific
              unsupported formats (e.g., WEBP/GIF edge cases).
            - Falls back to raw file bytes on conversion failure.
        """
        path = Path(image_path)
        suffix = path.suffix.lower()

        # PDF -> first page PNG bytes
        if suffix == ".pdf":
            try:
                import fitz  # pymupdf
                doc = fitz.open(str(path))
                try:
                    if len(doc) == 0:
                        return path.read_bytes()
                    page = doc[0]
                    # Slight upscaling for better VLM readability
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    return pix.tobytes("png")
                finally:
                    doc.close()
            except Exception as e:
                logger.warning("vlm_service.pdf_to_png_failed path=%s err=%s", image_path, e)
                return path.read_bytes()

        # Any raster image -> normalize to PNG
        try:
            from PIL import Image as PILImage
            with PILImage.open(str(path)) as img:
                if img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGB")
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                return buf.getvalue()
        except Exception as e:
            logger.warning("vlm_service.image_normalize_failed path=%s err=%s", image_path, e)
            return path.read_bytes()

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        """Parse JSON from VLM response text."""
        if not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                try:
                    return json.loads(text[start:end].strip())
                except json.JSONDecodeError:
                    pass
        elif "```" in text:
            start = text.find("```") + 3
            nl = text.find("\n", start)
            if nl > start:
                start = nl + 1
            end = text.find("```", start)
            if end > start:
                try:
                    return json.loads(text[start:end].strip())
                except json.JSONDecodeError:
                    pass
        return {}
