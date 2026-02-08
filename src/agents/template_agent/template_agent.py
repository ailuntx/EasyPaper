"""
Template Parser Agent
- **Description**:
    - Parses LaTeX template zip packages to extract format rules and constraints
    - Uses LLM to understand complex template structures
"""
from openai import AsyncOpenAI
from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated, Optional, IO
from langgraph.graph import StateGraph, START, END
import operator
from pathlib import Path
import json
import zipfile
import tempfile
import os
import re
from typing import List, Dict, Any
from fastapi import APIRouter
from ...config.schema import ModelConfig
from ..base import BaseAgent
from .router import create_template_router
from .models import TemplateInfo


TEMPLATE_ANALYSIS_PROMPT = """You are a LaTeX template analysis expert. 
Analyze the provided LaTeX template files and extract the following information:

1. **Document Class**: The document class being used (e.g., article, IEEEtran, acmart)
2. **Citation Style**: The citation command format (\cite, \citep, \citet, etc.)
3. **Figure Placement**: Default figure placement options (e.g., [htbp], [H])
4. **Required Packages**: List of required packages
5. **Bibliography Style**: The .bst file or biblatex style being used
6. **Column Format**: Single or double column layout
7. **Section Structure**: Available section commands and their hierarchy
8. **Has Abstract**: Whether the template includes an abstract section
9. **Has Acknowledgment**: Whether the template includes an acknowledgment section
10. **Template Structure**: Key placeholder sections in the template

Please analyze the following LaTeX content and respond in JSON format:
{
    "document_class": "string",
    "citation_style": "cite|citep|citet|other",
    "figure_placement": "htbp|H|other",
    "required_packages": ["list", "of", "packages"],
    "bib_style": "string or null",
    "column_format": "single|double",
    "section_commands": ["section", "subsection", ...],
    "has_abstract": true|false,
    "has_acknowledgment": true|false,
    "template_structure": {
        "sections": ["abstract", "introduction", ...],
        "placeholders": {"key": "description"}
    }
}

LaTeX Template Content:
<template>
{content}
</template>
"""


class TemplateAgentState(TypedDict):
    """
    State for Template Parser Agent workflow
    """
    messages: Annotated[list[AnyMessage], operator.add]
    file_path: Optional[str]
    file_content: Optional[IO[bytes]]
    template_id: Optional[str]
    extracted_files: Optional[Dict[str, str]]  # filename -> content
    main_tex_content: Optional[str]
    template_info: Optional[Dict[str, Any]]
    llm_calls: int


class TemplateParserAgent(BaseAgent):
    """
    Template Parser Agent for analyzing LaTeX templates
    - **Description**:
        - Extracts and analyzes LaTeX template zip packages
        - Uses LLM to understand template structure and requirements
    """
    
    def __init__(self, config: ModelConfig):
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )
        self.model_name = config.model_name
        self.agent = self.init_agent()

    def init_agent(self):
        """
        Initialize the agent workflow graph
        """
        agent_builder = StateGraph(TemplateAgentState)
        agent_builder.add_node("extract_template", self.extract_template)
        agent_builder.add_node("analyze_template", self.analyze_template)
        agent_builder.add_edge(START, "extract_template")
        agent_builder.add_edge("extract_template", "analyze_template")
        agent_builder.add_edge("analyze_template", END)
        return agent_builder.compile()

    async def extract_template(self, state: TemplateAgentState) -> Dict[str, Any]:
        """
        Extract files from the template zip package
        - **Description**:
            - Unzips the template and extracts all .tex, .cls, .sty, .bst files
            - Identifies the main .tex file

        - **Args**:
            - `state` (TemplateAgentState): Current workflow state

        - **Returns**:
            - `dict`: Updated state with extracted files
        """
        print(f"INPUT STATE [extract_template]: {state}")
        
        extracted_files = {}
        main_tex_content = None
        main_tex_path = None
        
        # Get the zip file path
        zip_path = state.get("file_path")
        if not zip_path:
            raise ValueError("file_path must be provided for template extraction")
        
        # Extract to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Walk through extracted files
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, temp_dir)
                    
                    # Extract relevant file types
                    if file.endswith(('.tex', '.cls', '.sty', '.bst', '.bib')):
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                                extracted_files[rel_path] = content
                                
                                # Identify main .tex file
                                if file.endswith('.tex'):
                                    if '\\documentclass' in content and '\\begin{document}' in content:
                                        if main_tex_content is None or file == 'main.tex':
                                            main_tex_content = content
                                            main_tex_path = rel_path
                        except Exception as e:
                            print(f"Warning: Could not read {file_path}: {e}")
        
        # If no main tex found, use the first .tex file
        if main_tex_content is None:
            for path, content in extracted_files.items():
                if path.endswith('.tex'):
                    main_tex_content = content
                    main_tex_path = path
                    break
        
        return {
            "extracted_files": extracted_files,
            "main_tex_content": main_tex_content,
            "template_id": state.get("template_id") or Path(zip_path).stem,
        }

    def _clean_json_content(self, content: str) -> str:
        """
        Clean and extract JSON from LLM response
        - **Description**:
            - Strips whitespace and attempts to extract valid JSON
            - Handles cases where LLM returns JSON wrapped in markdown or extra text

        - **Args**:
            - `content` (str): Raw LLM response content

        - **Returns**:
            - `str`: Cleaned JSON string
        """
        if not content:
            return "{}"
        
        # Strip whitespace
        content = content.strip()
        
        # If already starts with {, return as-is
        if content.startswith('{'):
            return content
        
        # Try to extract JSON block from markdown code fence
        markdown_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', content)
        if markdown_match:
            return markdown_match.group(1)
        
        # Try to find any JSON object in the content
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            return json_match.group()
        
        # Return empty object if no JSON found
        return "{}"

    async def analyze_template(self, state: TemplateAgentState) -> Dict[str, Any]:
        """
        Analyze the template using LLM
        - **Description**:
            - Sends template content to LLM for analysis
            - Extracts structure and format rules
            - Falls back to rule-based parsing if LLM fails

        - **Args**:
            - `state` (TemplateAgentState): Current workflow state

        - **Returns**:
            - `dict`: Updated state with template info
        """
        print(f"INPUT STATE [analyze_template]: {state}")
        
        main_content = state.get("main_tex_content", "")
        extracted_files = state.get("extracted_files", {})
        template_info = None
        
        if not main_content:
            # Try rule-based parsing if no content
            template_info = self._parse_template_rules(extracted_files)
            template_info["template_id"] = state.get("template_id", "unknown")
        else:
            # Combine relevant content for analysis
            combined_content = main_content
            
            # Add class/style file content if available
            for path, content in extracted_files.items():
                if path.endswith(('.cls', '.sty')) and len(content) < 5000:
                    combined_content += f"\n\n--- {path} ---\n{content[:3000]}"
            
            # Limit content size for LLM
            if len(combined_content) > 15000:
                combined_content = combined_content[:15000] + "\n... (truncated)"
            
            # Call LLM for analysis
            prompt = TEMPLATE_ANALYSIS_PROMPT.format(content=combined_content)
            
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a LaTeX template analysis expert. Always respond with valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={'type': 'json_object'},
                    temperature=0.1,
                )
                
                raw_content = response.choices[0].message.content
                print(f"LLM raw response (first 500 chars): {raw_content[:500] if raw_content else 'None'}")
                
                # Clean and parse JSON
                cleaned_content = self._clean_json_content(raw_content)
                result = json.loads(cleaned_content)
                
                # Build TemplateInfo
                template_info = {
                    "template_id": state.get("template_id", "unknown"),
                    "main_tex_path": self._find_main_tex_path(extracted_files),
                    "citation_style": result.get("citation_style", "cite"),
                    "figure_placement": result.get("figure_placement", "htbp"),
                    "section_commands": result.get("section_commands", ["section", "subsection"]),
                    "required_packages": result.get("required_packages", []),
                    "bib_style": result.get("bib_style"),
                    "document_class": result.get("document_class", "article"),
                    "template_structure": result.get("template_structure", {}),
                    "has_abstract": result.get("has_abstract", True),
                    "has_acknowledgment": result.get("has_acknowledgment", False),
                    "column_format": result.get("column_format", "single"),
                    "raw_preamble": self._extract_preamble(main_content),
                }
                print(f"LLM analysis successful: document_class={template_info.get('document_class')}")
                
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed: {e}, falling back to rule-based parsing")
                template_info = self._parse_template_rules(extracted_files)
                template_info["template_id"] = state.get("template_id", "unknown")
                template_info["raw_preamble"] = self._extract_preamble(main_content)
                
            except Exception as e:
                print(f"LLM analysis failed: {e}, falling back to rule-based parsing")
                template_info = self._parse_template_rules(extracted_files)
                template_info["template_id"] = state.get("template_id", "unknown")
                template_info["raw_preamble"] = self._extract_preamble(main_content)
        
        return {"template_info": template_info}

    def _find_main_tex_path(self, extracted_files: Dict[str, str]) -> str:
        """Find the path to the main .tex file"""
        for path, content in extracted_files.items():
            if path.endswith('.tex') and '\\documentclass' in content:
                return path
        return "main.tex"

    def _extract_preamble(self, content: str) -> Optional[str]:
        """Extract the preamble from LaTeX content"""
        if not content:
            return None
        
        match = re.search(r'(\\documentclass.*?)\\begin\{document\}', content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _parse_template_rules(self, extracted_files: Dict[str, str]) -> Dict[str, Any]:
        """
        Rule-based template parsing fallback
        - **Description**:
            - Parses template using regex patterns when LLM is not available
        """
        template_info = {
            "template_id": "unknown",
            "main_tex_path": "main.tex",
            "citation_style": "cite",
            "figure_placement": "htbp",
            "section_commands": ["section", "subsection", "subsubsection"],
            "required_packages": [],
            "bib_style": None,
            "document_class": "article",
            "template_structure": {},
            "has_abstract": True,
            "has_acknowledgment": False,
            "column_format": "single",
        }
        
        for path, content in extracted_files.items():
            if not path.endswith('.tex'):
                continue
                
            # Extract document class
            doc_class = re.search(r'\\documentclass(?:\[.*?\])?\{(\w+)\}', content)
            if doc_class:
                template_info["document_class"] = doc_class.group(1)
            
            # Check for two-column
            if 'twocolumn' in content or 'IEEEtran' in content:
                template_info["column_format"] = "double"
            
            # Extract packages
            packages = re.findall(r'\\usepackage(?:\[.*?\])?\{([^}]+)\}', content)
            for pkg in packages:
                for p in pkg.split(','):
                    p = p.strip()
                    if p and p not in template_info["required_packages"]:
                        template_info["required_packages"].append(p)
            
            # Check citation style
            if '\\citep' in content or 'natbib' in content:
                template_info["citation_style"] = "citep"
            elif '\\citet' in content:
                template_info["citation_style"] = "citet"
            
            # Check for abstract
            if '\\begin{abstract}' in content or '\\abstract' in content:
                template_info["has_abstract"] = True
            
            # Check for acknowledgment
            if 'acknowledgment' in content.lower() or 'acknowledgement' in content.lower():
                template_info["has_acknowledgment"] = True
            
            # Extract bib style
            bib_style = re.search(r'\\bibliographystyle\{(\w+)\}', content)
            if bib_style:
                template_info["bib_style"] = bib_style.group(1)
        
        return template_info

    async def run(self, file_path: Optional[str] = None, 
                  file_content: Optional[IO[bytes]] = None,
                  template_id: Optional[str] = None):
        """
        Run the template parser agent
        - **Args**:
            - `file_path` (str, optional): Path to the template zip file
            - `file_content` (IO[bytes], optional): File content bytes
            - `template_id` (str, optional): Custom template ID

        - **Returns**:
            - `dict`: Parsed template information
        """
        return await self.agent.ainvoke({
            "file_path": file_path,
            "file_content": file_content,
            "template_id": template_id,
            "messages": [],
            "llm_calls": 0,
        })

    @property
    def name(self) -> str:
        """Agent name identifier"""
        return "template_parser"

    @property
    def description(self) -> str:
        """Agent description"""
        return "Parses LaTeX template packages to extract format rules and structure"

    @property
    def router(self) -> APIRouter:
        """Return the FastAPI router for this agent"""
        return create_template_router(self)

    @property
    def endpoints_info(self) -> List[Dict[str, Any]]:
        """Return endpoint metadata for list_agents"""
        return [
            {
                "path": "/agent/template/parse",
                "method": "POST",
                "description": "Parse LaTeX template zip and extract format rules",
                "input_model": "TemplateParsePayload",
                "output_model": "TemplateParseResult"
            }
        ]
