from fastapi import APIRouter, HTTPException, status
from typing import Optional, Any, Dict, List
import time
import logging
from pydantic import BaseModel, Field
from .models import ParsePayload, ParseResult


class PaperSearchRequest(BaseModel):
    query: str
    max_results: int = Field(default=5, ge=1, le=20)
    source: str = Field(default="semantic_scholar", pattern="^(semantic_scholar|arxiv)$")
    year_range: Optional[str] = None
    query_field: str = Field(default="all", pattern="^(all|ti|abs|au|cat)$")


class PaperSearchResponse(BaseModel):
    papers: List[Dict[str, Any]]
    bibtex: str
    total_found: int


def create_parse_router(agent_instance):
    """Create router for parse agent endpoints"""
    router = APIRouter()
    logger = logging.getLogger("uvicorn.error")

    @router.post("/agent/parse", response_model=ParseResult, status_code=status.HTTP_200_OK)
    async def parse_paper(payload: ParsePayload):
        """Parse and understand a research paper using the paper parser agent"""
        # basic metrics/logging
        start = time.time()
        logger.info("parse.request %s user=%s", payload.request_id, payload.user_id)

        try:
            # Extract file information from payload
            file_path = payload.payload.get("file_path")
            file_content = payload.payload.get("file_content")

            if not file_path and not file_content:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Either file_path or file_content must be provided"
                )

            # Run the agent
            agent_result = await agent_instance.run(file_path=file_path, file_content=file_content)

            # Extract the understanding result
            understand_result = agent_result.get("understand_result", {})

            latency = time.time() - start
            logger.info("parse.complete %s latency=%.3f", payload.request_id, latency)

            return ParseResult(
                request_id=payload.request_id,
                status="ok",
                result=understand_result
            )

        except Exception as e:
            latency = time.time() - start
            logger.error("parse.error %s latency=%.3f error=%s", payload.request_id, latency, str(e))
            return ParseResult(
                request_id=payload.request_id,
                status="error",
                error=str(e)
            )

    @router.post(
        "/agent/papers/search",
        response_model=PaperSearchResponse,
        status_code=status.HTTP_200_OK,
    )
    async def search_papers(req: PaperSearchRequest):
        """
        Search academic papers via Semantic Scholar or arXiv.
        - **Description**:
            - Instantiates PaperSearchTool and runs a search query.
            - Returns paper metadata with BibTeX entries.
        """
        from ..shared.tools.paper_search import PaperSearchTool

        tool = PaperSearchTool()
        start = time.time()
        logger.info("papers.search query=%r source=%s max=%d", req.query, req.source, req.max_results)

        try:
            result = await tool.execute(
                query=req.query,
                max_results=req.max_results,
                source=req.source,
                year_range=req.year_range,
                query_field=req.query_field,
            )
            latency = time.time() - start
            logger.info("papers.search.complete latency=%.3f found=%d", latency, result.data.get("total_found", 0))

            return PaperSearchResponse(
                papers=result.data.get("papers", []),
                bibtex=result.data.get("bibtex", ""),
                total_found=result.data.get("total_found", 0),
            )
        except Exception as e:
            latency = time.time() - start
            logger.error("papers.search.error latency=%.3f error=%s", latency, str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Paper search failed: {str(e)}",
            )

    return router