from openai import AsyncOpenAI
from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated, Optional, IO
from langgraph.graph import StateGraph, START, END
import operator
from pathlib import Path
import json
from typing import List, Dict, Any
from fastapi import APIRouter
from ...config.schema import ModelConfig
from ..base import BaseAgent
from .router import create_parse_router


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

class ParseAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    file_path: Optional[str] = None
    file_content: Optional[IO[bytes]] = None
    file_id: Optional[str] = None
    understand_result: Optional[dict] = None
    llm_calls: int

class ParseAgent(BaseAgent):
    def __init__(self, config: ModelConfig):
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )
        self.model_name = config.model_name
        self.agent = self.init_agent()

    def init_agent(self):
        agent_builder = StateGraph(ParseAgentState)
        agent_builder.add_node("upload_file", self.upload_file)
        agent_builder.add_node("understand_paper", self.understand_paper)
        agent_builder.add_edge(START, "upload_file")
        agent_builder.add_edge("upload_file", "understand_paper")
        agent_builder.add_edge("understand_paper", END)
        return agent_builder.compile()

    async def upload_file(self, state: ParseAgentState):
        """Upload a file to the model"""
        print(f"INPUT STATE [upload_file]: {state}")
        if state["file_path"]:
            file_obj = await self.client.files.create(file=Path(state["file_path"]), purpose="file-extract")
        elif state["file_content"]:
            file_obj = await self.client.files.create(file=state["file_content"], purpose="file-extract")
        else:
            raise ValueError("Either file_path or file_content must be provided")
        state["file_id"] = file_obj.id
        return state

    async def understand_paper(self, state: ParseAgentState):
        """Understand the paper"""
        print(f"INPUT STATE [understand_paper]: {state}")
        if state["file_id"]:
            file_obj = await self.client.files.content(file_id=state["file_id"])
            file_obj = file_obj.text
            file_content = json.loads(file_obj)['content']
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": UNDERSTAND_PROMPT},
                    {"role": "user", "content": f"<paper>{file_content}</paper>"}
                ],
                response_format={'type': 'json_object'}
            )
            await self.client.files.delete(file_id=state["file_id"])
        else:
            raise ValueError("File ID must be provided")
        return {
            "understand_result": json.loads(response.choices[0].message.content)
        }

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
    def router(self) -> APIRouter:
        """Return the FastAPI router for this agent"""
        return create_parse_router(self)

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

async def main():
    config = ModelConfig(
        model_name="openai/gpt-4o-mini",
        api_key = "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",
        base_url = "https://openrouter.ai/api/v1",
    )
    agent = ParseAgent(config)
    result = await agent.run(file_path="./test_pdf.pdf")
    print(result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())