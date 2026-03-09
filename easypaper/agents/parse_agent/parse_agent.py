from langchain.messages import AnyMessage
from ..shared.llm_client import LLMClient
from typing_extensions import TypedDict, Annotated, Optional, IO
from langgraph.graph import StateGraph, START, END
import asyncio
import operator
from pathlib import Path
import json
from typing import List, Dict, Any
from ...config.schema import ModelConfig
from ..base import BaseAgent


UNDERSTAND_PROMPT = """You are a helpful assistant that helps to understand paper.
User will provide you the raw contents of the paper within a <paper> tag. 
Please use critical thinking to understand the paper and provide a summary of the paper.

Please summarize the paper from following aspects:
- Summary (str): a brief summary of the paper
- Research Background (str): the background of the research
- Research Question (str): the question of the research
- Research Hypothesis (list[str]): the hypothesis of the research
- Methods (list[str]): the methods of the research
- Results (list[str]): the results of the research
- Key Findings (list[str]): the key findings of the research

Please output in JSON format.
{
    "summary": "...",
    "research_background": "...",
    "research_question": "...",
    "research_hypothesis": [...],
    "methods": [...],
    "results": [...],
    "key_findings": [...],
}
"""

MAX_TEXT_CHARS = 48_000
MAX_RETRIES = 3
RETRY_BASE_WAIT = 3


def _extract_pdf_text(path: str) -> str:
    """
    Extract text from a PDF using PyMuPDF (fitz).
    - **Description**:
        - Falls back gracefully if the file cannot be read.
        - Truncates to MAX_TEXT_CHARS to stay within LLM context limits.

    - **Args**:
        - `path` (str): Absolute path to the PDF file.

    - **Returns**:
        - `str`: Extracted text content.
    """
    import fitz
    doc = fitz.open(path)
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    text = "\n\n".join(pages)
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS] + "\n... [truncated]"
    return text


def _extract_bytes_text(content: IO[bytes]) -> str:
    """
    Extract text from in-memory PDF bytes using PyMuPDF.

    - **Args**:
        - `content` (IO[bytes]): File-like object with PDF bytes.

    - **Returns**:
        - `str`: Extracted text content.
    """
    import fitz
    raw = content.read() if hasattr(content, "read") else content
    doc = fitz.open(stream=raw, filetype="pdf")
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    text = "\n\n".join(pages)
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS] + "\n... [truncated]"
    return text


class ParseAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    file_path: Optional[str] = None
    file_content: Optional[IO[bytes]] = None
    paper_text: Optional[str] = None
    understand_result: Optional[dict] = None
    llm_calls: int

class ParseAgent(BaseAgent):
    def __init__(self, config: ModelConfig):
        self.client = LLMClient(
            api_key=config.api_key,
            base_url=config.base_url,
        )
        self.model_name = config.model_name
        self.agent = self.init_agent()

    def init_agent(self):
        agent_builder = StateGraph(ParseAgentState)
        agent_builder.add_node("extract_text", self.extract_text)
        agent_builder.add_node("understand_paper", self.understand_paper)
        agent_builder.add_edge(START, "extract_text")
        agent_builder.add_edge("extract_text", "understand_paper")
        agent_builder.add_edge("understand_paper", END)
        return agent_builder.compile()

    async def extract_text(self, state: ParseAgentState):
        """Extract text from PDF locally using PyMuPDF instead of remote Files API."""
        print(f"INPUT STATE [extract_text]: file_path={state.get('file_path')}, "
              f"has_content={state.get('file_content') is not None}")

        if state.get("file_path"):
            text = _extract_pdf_text(state["file_path"])
        elif state.get("file_content"):
            text = _extract_bytes_text(state["file_content"])
        else:
            raise ValueError("Either file_path or file_content must be provided")

        print(f"[extract_text] Extracted {len(text)} chars from PDF")
        return {"paper_text": text}

    async def understand_paper(self, state: ParseAgentState):
        """
        Understand the paper by sending extracted text to the LLM.
        - **Description**:
            - Retries up to MAX_RETRIES times with exponential backoff on
              transient server errors (busy, 429, 5xx).
        """
        paper_text = state.get("paper_text")
        if not paper_text:
            raise ValueError("No paper text available for analysis")

        print(f"INPUT STATE [understand_paper]: text_len={len(paper_text)}")

        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": UNDERSTAND_PROMPT},
                        {"role": "user", "content": f"<paper>{paper_text}</paper>"}
                    ],
                    response_format={'type': 'json_object'}
                )
                return {
                    "understand_result": json.loads(response.choices[0].message.content)
                }
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                is_transient = any(k in err_str for k in ("busy", "retry", "429", "500", "502", "503", "overloaded"))
                if is_transient and attempt < MAX_RETRIES:
                    wait = RETRY_BASE_WAIT * (2 ** (attempt - 1))
                    print(f"[understand_paper] Transient error (attempt {attempt}/{MAX_RETRIES}), "
                          f"retrying in {wait}s: {e}")
                    await asyncio.sleep(wait)
                    continue
                raise

        raise last_error  # type: ignore[misc]

    async def run(self, file_path: Optional[str] = None, file_content: Optional[IO[bytes]] = None):
        """Run the agent"""
        return await self.agent.ainvoke({
            "file_path": file_path,
            "file_content": file_content,
        })

    @property
    def name(self) -> str:
        """Agent name identifier"""
        return "paper_parser"

    @property
    def description(self) -> str:
        """Agent description"""
        return "Research paper understanding and parsing agent"

    @property
    def router(self) -> "APIRouter | None":
        """Return the FastAPI router for this agent"""
        try:
            from .router import create_parse_router
            return create_parse_router(self)
        except Exception:
            return None

    @property
    def endpoints_info(self) -> List[Dict[str, Any]]:
        """Return endpoint metadata for list_agents"""
        return [
            {
                "path": "/agent/parse",
                "method": "POST",
                "description": "Parse research paper and extract structured information",
                "input_model": "ParsePayload",
                "output_model": "ParseResult"
            }
        ]
