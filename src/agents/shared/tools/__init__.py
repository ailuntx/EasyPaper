from .base import WriterTool, ToolResult
from .registry import ToolRegistry
from .citation_tools import CitationValidatorTool, WordCountTool, KeyPointCoverageTool
from .paper_search import PaperSearchTool

TOOL_FACTORY = {
    "validate_citations": CitationValidatorTool,
    "count_words": WordCountTool,
    "check_key_points": KeyPointCoverageTool,
    "search_papers": PaperSearchTool,
}

__all__ = [
    "WriterTool",
    "ToolResult",
    "ToolRegistry",
    "CitationValidatorTool",
    "WordCountTool",
    "KeyPointCoverageTool",
    "PaperSearchTool",
    "TOOL_FACTORY",
]
