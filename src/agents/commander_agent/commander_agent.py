"""
Commander Agent
- **Description**:
    - Orchestrates paper writing workflow by assembling context from FlowGram.ai
    - Converts graph nodes to argument tree structure (claims + materials)
    - Outputs unified SectionWritePayload for Writer Agent
    - Acts as the adapter between FlowGram.ai and the Writer Agent
"""
from openai import AsyncOpenAI
from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated, Optional
from langgraph.graph import StateGraph, START, END
import operator
import json
import httpx
from typing import List, Dict, Any
from fastapi import APIRouter
from ...config.schema import ModelConfig
from ..base import BaseAgent
from .router import create_commander_router
# Import unified models from Writer Agent
from ..writer_agent.section_models import (
    Material,
    Point,
    ArgumentStructure,
    ReferenceInfo,
    FigureInfo,
    TableInfo,
    EquationInfo,
    TemplateRules,
    SectionResources,
    SectionConstraints,
    SectionWritePayload,
)


# Node types that typically represent claims/assertions
CLAIM_NODE_TYPES = {"hypothesis", "question", "idea", "finding"}

# Node types that typically represent supporting evidence
EVIDENCE_NODE_TYPES = {"method", "experiment", "result", "data", "metric", "observation"}

# Node types that typically represent background context
BACKGROUND_NODE_TYPES = {"literature", "concept"}


# Backend API URL for graph data
BACKEND_API_URL = "http://127.0.0.1:9001/api"


SECTION_PROMPTS = {
    "abstract": """You are writing the Abstract section of a research paper.
The abstract should:
- Summarize the research problem and motivation (1-2 sentences)
- Describe the methodology briefly (1-2 sentences)
- Present key results and findings (1-2 sentences)
- State the main conclusions and implications (1-2 sentences)
Keep it concise, typically 150-250 words.""",

    "introduction": """You are writing the Introduction section of a research paper.
The introduction should:
- Establish the research context and background
- Identify the problem or gap in current knowledge
- State the research objectives and contributions
- Outline the paper structure
Use a clear narrative flow from general to specific.""",

    "related_work": """You are writing the Related Work section of a research paper.
This section should:
- Survey relevant prior work systematically
- Group related works by theme or approach
- Identify gaps that your work addresses
- Clearly differentiate your contribution from existing work
Use proper citations throughout.""",

    "method": """You are writing the Method/Methodology section of a research paper.
This section should:
- Describe your approach in sufficient detail for reproduction
- Explain the rationale behind methodological choices
- Include formal definitions, algorithms, or models as needed
- Use clear notation and terminology consistently.""",

    "experiment": """You are writing the Experiment section of a research paper.
This section should:
- Describe the experimental setup and configuration
- Specify datasets, metrics, and baselines used
- Explain evaluation protocols and procedures
- Provide implementation details as necessary.""",

    "result": """You are writing the Results section of a research paper.
This section should:
- Present experimental results clearly and objectively
- Use tables and figures to support key findings
- Compare against baselines and prior work
- Highlight statistically significant results.""",

    "discussion": """You are writing the Discussion section of a research paper.
This section should:
- Interpret the results in context of research questions
- Discuss implications and significance of findings
- Address limitations and potential threats to validity
- Suggest directions for future work.""",

    "conclusion": """You are writing the Conclusion section of a research paper.
This section should:
- Summarize the main contributions concisely
- Restate key findings and their significance
- Discuss broader impact and applications
- End with forward-looking perspective.""",
}


class CommanderAgentState(TypedDict):
    """
    State for Commander Agent workflow
    - **Description**:
        - Internal state for the Commander workflow
        - Final output is SectionWritePayload for Writer Agent
    """
    messages: Annotated[list[AnyMessage], operator.add]
    # Input parameters
    work_id: Optional[str]
    section_type: Optional[str]
    section_title: Optional[str]
    user_prompt: Optional[str]
    template_id: Optional[str]
    explicit_node_ids: Optional[List[str]]
    word_count_limit: Optional[int]
    # Fetched data
    graph_data: Optional[Dict[str, Any]]
    raw_references: Optional[List[Dict[str, Any]]]
    template_rules: Optional[Dict[str, Any]]
    # Extracted context (will be converted to unified models)
    explicit_nodes: Optional[List[Dict[str, Any]]]
    implicit_nodes: Optional[List[Dict[str, Any]]]
    figures: Optional[List[Dict[str, Any]]]
    tables: Optional[List[Dict[str, Any]]]
    equations: Optional[List[Dict[str, Any]]]
    references: Optional[List[Dict[str, Any]]]
    # Final output
    section_write_payload: Optional[SectionWritePayload]
    llm_calls: int


class CommanderAgent(BaseAgent):
    """
    Commander Agent for orchestrating paper writing
    - **Description**:
        - Assembles context from research graph
        - Compiles structured prompts for Writer Agent
        - Handles context augmentation for implicit dependencies
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
        agent_builder = StateGraph(CommanderAgentState)
        agent_builder.add_node("fetch_graph_data", self.fetch_graph_data)
        agent_builder.add_node("extract_explicit_context", self.extract_explicit_context)
        agent_builder.add_node("augment_implicit_context", self.augment_implicit_context)
        agent_builder.add_node("extract_visual_resources", self.extract_visual_resources)
        agent_builder.add_node("compile_writer_prompt", self.compile_writer_prompt)
        
        agent_builder.add_edge(START, "fetch_graph_data")
        agent_builder.add_edge("fetch_graph_data", "extract_explicit_context")
        agent_builder.add_edge("extract_explicit_context", "augment_implicit_context")
        agent_builder.add_edge("augment_implicit_context", "extract_visual_resources")
        agent_builder.add_edge("extract_visual_resources", "compile_writer_prompt")
        agent_builder.add_edge("compile_writer_prompt", END)
        
        return agent_builder.compile()

    async def fetch_graph_data(self, state: CommanderAgentState) -> Dict[str, Any]:
        """
        Fetch research graph data from backend
        - **Description**:
            - Retrieves work data including nodes, edges, and references

        - **Args**:
            - `state` (CommanderAgentState): Current workflow state

        - **Returns**:
            - `dict`: Updated state with graph data
        """
        print(f"INPUT STATE [fetch_graph_data]: work_id={state.get('work_id')}")
        
        work_id = state.get("work_id")
        if not work_id:
            raise ValueError("work_id must be provided")
        
        graph_data = {}
        references = []
        
        try:
            async with httpx.AsyncClient() as client:
                # Fetch work and graph data
                work_response = await client.get(
                    f"{BACKEND_API_URL}/works/{work_id}",
                    timeout=30.0
                )
                if work_response.status_code == 200:
                    graph_data = work_response.json()
                
                # Fetch references
                refs_response = await client.get(
                    f"{BACKEND_API_URL}/references",
                    timeout=30.0
                )
                if refs_response.status_code == 200:
                    references = refs_response.json()
        except Exception as e:
            print(f"Warning: Could not fetch graph data: {e}")
        
        return {
            "graph_data": graph_data,
            "references": references,
        }

    async def extract_explicit_context(self, state: CommanderAgentState) -> Dict[str, Any]:
        """
        Extract context from explicitly referenced nodes
        - **Description**:
            - Processes nodes that user explicitly referenced
            - Converts node data to structured context

        - **Args**:
            - `state` (CommanderAgentState): Current workflow state

        - **Returns**:
            - `dict`: Updated state with explicit node contexts
        """
        print(f"INPUT STATE [extract_explicit_context]: explicit_ids={state.get('explicit_node_ids')}")
        
        graph_data = state.get("graph_data", {})
        explicit_ids = state.get("explicit_node_ids", [])
        nodes = graph_data.get("nodes", [])
        
        explicit_nodes = []
        
        # Build node map for quick lookup
        node_map = self._build_node_map(nodes)
        
        # Extract context from explicit nodes
        if explicit_ids:
            for node_id in explicit_ids:
                if node_id in node_map:
                    node = node_map[node_id]
                    context = self._node_to_context(node)
                    explicit_nodes.append(context)
        else:
            # If no explicit nodes specified, use section-relevant nodes
            section_type = state.get("section_type", "")
            relevant_nodes = self._find_relevant_nodes(nodes, section_type)
            for node in relevant_nodes:
                context = self._node_to_context(node)
                explicit_nodes.append(context)
        
        return {"explicit_nodes": explicit_nodes}

    async def augment_implicit_context(self, state: CommanderAgentState) -> Dict[str, Any]:
        """
        Augment context with implicit dependencies
        - **Description**:
            - Finds upstream/downstream nodes that provide necessary context
            - Prevents hallucination by including supporting evidence

        - **Args**:
            - `state` (CommanderAgentState): Current workflow state

        - **Returns**:
            - `dict`: Updated state with implicit node contexts
        """
        print(f"INPUT STATE [augment_implicit_context]")
        
        graph_data = state.get("graph_data", {})
        explicit_nodes = state.get("explicit_nodes", [])
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])
        
        # Build edge maps
        node_map = self._build_node_map(nodes)
        incoming_edges = {}  # target -> [sources]
        outgoing_edges = {}  # source -> [targets]
        
        for edge in edges:
            source = edge.get("sourceNodeID")
            target = edge.get("targetNodeID")
            if source and target:
                incoming_edges.setdefault(target, []).append(source)
                outgoing_edges.setdefault(source, []).append(target)
        
        # Find implicit nodes (upstream dependencies)
        explicit_ids = {n.get("node_id") for n in explicit_nodes}
        implicit_ids = set()
        
        for node in explicit_nodes:
            # Add upstream nodes (dependencies)
            node_id = node.get("node_id")
            for source_id in incoming_edges.get(node_id, []):
                if source_id not in explicit_ids:
                    implicit_ids.add(source_id)
        
        # Convert implicit nodes to context
        implicit_nodes = []
        for node_id in implicit_ids:
            if node_id in node_map:
                context = self._node_to_context(node_map[node_id])
                implicit_nodes.append(context)
        
        return {"implicit_nodes": implicit_nodes}

    async def extract_visual_resources(self, state: CommanderAgentState) -> Dict[str, Any]:
        """
        Extract figures, tables, and equations from context nodes
        - **Description**:
            - Scans explicit and implicit nodes for visual resources
            - Extracts figure files, table data, and equation definitions

        - **Args**:
            - `state` (CommanderAgentState): Current workflow state

        - **Returns**:
            - `dict`: Updated state with figures, tables, and equations
        """
        print(f"INPUT STATE [extract_visual_resources]")
        
        graph_data = state.get("graph_data", {})
        nodes = graph_data.get("nodes", [])
        node_map = self._build_node_map(nodes)
        
        figures: List[Dict[str, Any]] = []
        tables: List[Dict[str, Any]] = []
        equations: List[Dict[str, Any]] = []
        
        # Collect all node IDs to scan
        explicit_nodes = state.get("explicit_nodes", [])
        implicit_nodes = state.get("implicit_nodes", [])
        all_node_ids = set()
        
        for n in explicit_nodes:
            all_node_ids.add(n.get("node_id"))
        for n in implicit_nodes:
            all_node_ids.add(n.get("node_id"))
        
        # Also scan all nodes for figures/tables that might be relevant
        for node_id, node in node_map.items():
            node_type = node.get("type", "")
            data = node.get("data", {})
            inputs = data.get("inputsValues", {})
            
            if node_type == "figure":
                # Extract figure information
                figure_info = {
                    "figure_id": node_id,
                    "node_id": node_id,
                    "title": data.get("title", ""),
                    "caption": inputs.get("Caption", {}).get("content", ""),
                    "file_path": inputs.get("FilePath", {}).get("content", ""),
                }
                figures.append(figure_info)
                
            elif node_type == "table":
                # Extract table information
                table_data = inputs.get("Data", {}).get("content", None)
                table_latex = inputs.get("LaTeXContent", {}).get("content", "")
                
                table_info = {
                    "table_id": node_id,
                    "node_id": node_id,
                    "title": data.get("title", ""),
                    "caption": inputs.get("Caption", {}).get("content", ""),
                    "latex_content": table_latex,
                }
                if table_data:
                    try:
                        table_info["data"] = json.loads(table_data) if isinstance(table_data, str) else table_data
                    except (json.JSONDecodeError, TypeError):
                        pass
                tables.append(table_info)
                
            elif node_type == "method":
                # Extract equations from method nodes
                equation_content = inputs.get("Equation", {}).get("content", "")
                if equation_content:
                    equation_info = {
                        "equation_id": f"eq_{node_id}",
                        "node_id": node_id,
                        "title": data.get("title", ""),
                        "latex": equation_content,
                        "description": inputs.get("Description", {}).get("content", "")[:200],
                    }
                    equations.append(equation_info)
        
        return {
            "figures": figures,
            "tables": tables,
            "equations": equations,
        }

    async def compile_section_payload(self, state: CommanderAgentState) -> Dict[str, Any]:
        """
        Compile the unified SectionWritePayload for Writer Agent
        - **Description**:
            - Converts extracted nodes into argument tree structure
            - Infers claims from hypothesis/question nodes
            - Organizes other nodes as supporting materials
            - This is the interface between Commander and Writer

        - **Args**:
            - `state` (CommanderAgentState): Current workflow state

        - **Returns**:
            - `dict`: Updated state with section_write_payload
        """
        print(f"INPUT STATE [compile_section_payload]")
        
        section_type = state.get("section_type", "introduction")
        section_title = state.get("section_title", "")
        user_prompt = state.get("user_prompt", "")
        word_count_limit = state.get("word_count_limit")
        template_rules_raw = state.get("template_rules", {})
        
        # Collect all nodes
        explicit_nodes_raw = state.get("explicit_nodes", [])
        implicit_nodes_raw = state.get("implicit_nodes", [])
        all_nodes_raw = explicit_nodes_raw + implicit_nodes_raw
        
        # Convert raw nodes to Materials
        all_materials = [
            Material(
                id=n.get("node_id", ""),
                material_type=n.get("node_type", ""),
                title=n.get("title", ""),
                content=n.get("content", ""),
                metadata=n.get("metadata", {}),
            )
            for n in all_nodes_raw
        ]
        
        # Build argument structure by inferring claims from nodes
        argument = self._build_argument_structure(all_materials, section_type, user_prompt)
        
        # Convert references to ReferenceInfo
        references_raw = state.get("references", [])
        references = [
            ReferenceInfo(
                ref_id=r.get("id", ""),
                title=r.get("title", ""),
                authors=r.get("authors"),
                year=r.get("year"),
                venue=r.get("venue"),
                doi=r.get("doi"),
                url=r.get("url"),
                abstract=r.get("abstract"),
                bibtex=r.get("bibtex"),
            )
            for r in references_raw[:20]
        ]
        
        # Convert figures to FigureInfo
        figures_raw = state.get("figures", [])
        figures = [
            FigureInfo(
                figure_id=f.get("figure_id", ""),
                title=f.get("title", ""),
                caption=f.get("caption", ""),
                file_path=f.get("file_path"),
                file_type=f.get("file_type"),
                width=f.get("width"),
            )
            for f in figures_raw
        ]
        
        # Convert tables to TableInfo
        tables_raw = state.get("tables", [])
        tables = [
            TableInfo(
                table_id=t.get("table_id", ""),
                title=t.get("title", ""),
                caption=t.get("caption", ""),
                data=t.get("data"),
                latex_content=t.get("latex_content"),
            )
            for t in tables_raw
        ]
        
        # Convert equations to EquationInfo
        equations_raw = state.get("equations", [])
        equations = [
            EquationInfo(
                equation_id=e.get("equation_id", ""),
                title=e.get("title", ""),
                latex=e.get("latex", ""),
                description=e.get("description"),
            )
            for e in equations_raw
        ]
        
        # Build TemplateRules
        template_rules = None
        if template_rules_raw:
            template_rules = TemplateRules(
                document_class=template_rules_raw.get("document_class"),
                citation_style=template_rules_raw.get("citation_style", "cite"),
                figure_placement=template_rules_raw.get("figure_placement"),
                column_format=template_rules_raw.get("column_format"),
                section_commands=template_rules_raw.get("section_commands", []),
                required_packages=template_rules_raw.get("required_packages", []),
            )
        
        # Build SectionResources
        resources = SectionResources(
            references=references,
            figures=figures,
            tables=tables,
            equations=equations,
            template_rules=template_rules,
        )
        
        # Build SectionConstraints
        citation_style = template_rules_raw.get("citation_style", "cite") if template_rules_raw else "cite"
        section_constraints = SectionConstraints(
            word_count_limit=word_count_limit,
            citation_format=citation_style,
            language="en",
            style_guide=template_rules_raw.get("style_guide") if template_rules_raw else None,
        )
        
        # Build the unified SectionWritePayload
        payload = SectionWritePayload(
            section_type=section_type,
            section_title=section_title or section_type.replace("_", " ").title(),
            user_prompt=user_prompt or f"Write the {section_type} section based on the provided context.",
            argument=argument,
            resources=resources,
            constraints=section_constraints,
        )
        
        return {
            "section_write_payload": payload,
        }

    # Keep old method name as alias for compatibility
    async def compile_writer_prompt(self, state: CommanderAgentState) -> Dict[str, Any]:
        """Alias for compile_section_payload for backward compatibility"""
        return await self.compile_section_payload(state)

    def _build_argument_structure(
        self,
        materials: List[Material],
        section_type: str,
        user_prompt: str
    ) -> ArgumentStructure:
        """
        Build an argument structure from materials
        - **Description**:
            - Infers points from hypothesis/question/finding nodes
            - Assigns other nodes as supporting materials
            - Groups background nodes separately

        - **Args**:
            - `materials` (List[Material]): All extracted materials
            - `section_type` (str): The section type being written
            - `user_prompt` (str): User's writing instructions

        - **Returns**:
            - `ArgumentStructure`: The constructed argument tree
        """
        # Separate materials by role
        point_materials = []
        evidence_materials = []
        background_materials = []
        
        for m in materials:
            if m.material_type in CLAIM_NODE_TYPES:
                point_materials.append(m)
            elif m.material_type in BACKGROUND_NODE_TYPES:
                background_materials.append(m)
            else:
                evidence_materials.append(m)
        
        # Build points from point-type materials
        main_points = []
        
        if point_materials:
            # Each point-type material becomes a main point
            for pm in point_materials:
                # Find supporting materials for this point
                # Simple heuristic: assign evidence materials to points
                supporting = []
                
                # For now, distribute evidence materials among points
                # A more sophisticated approach would use graph edges
                point_index = point_materials.index(pm)
                chunk_size = max(1, len(evidence_materials) // len(point_materials))
                start_idx = point_index * chunk_size
                end_idx = start_idx + chunk_size if point_index < len(point_materials) - 1 else len(evidence_materials)
                supporting = evidence_materials[start_idx:end_idx]
                
                point = Point(
                    id=f"point-{pm.id}",
                    statement=pm.content[:200] if pm.content else pm.title,
                    point_type="main",
                    supporting_materials=supporting,
                    counter_materials=[],
                    sub_points=[],
                )
                main_points.append(point)
        else:
            # No explicit points found - create a default point from user prompt
            # and use all evidence materials as support
            default_point = Point(
                id="point-default",
                statement=user_prompt or f"Present the {section_type} content",
                point_type="main",
                supporting_materials=evidence_materials,
                counter_materials=[],
                sub_points=[],
            )
            main_points.append(default_point)
        
        # Generate thesis from section type and first point
        thesis = None
        if main_points:
            first_point = main_points[0].statement
            thesis = f"This section presents: {first_point[:150]}..."
        
        return ArgumentStructure(
            thesis=thesis,
            main_points=main_points,
            background_context=background_materials,
        )

    def _build_node_map(self, nodes: List[Dict]) -> Dict[str, Dict]:
        """Build a map of node_id -> node for quick lookup"""
        node_map = {}
        
        def add_node(node):
            node_map[node["id"]] = node
            for block in node.get("blocks", []):
                add_node(block)
        
        for node in nodes:
            add_node(node)
        
        return node_map

    def _node_to_context(self, node: Dict) -> Dict[str, Any]:
        """
        Convert a graph node to a dict (for Material conversion)
        - **Description**:
            - Extracts content from various node types
            - Handles different input field names per node type
            - Returns dict format compatible with Material model
        """
        data = node.get("data", {})
        inputs = data.get("inputsValues", {})
        
        # Extract content based on node type
        content = None
        node_type = node.get("type", "unknown")
        
        # Content extraction mapping by node type
        if node_type in ["idea", "hypothesis", "question"]:
            content = inputs.get("Description", {}).get("content", "")
        elif node_type == "method":
            desc = inputs.get("Description", {}).get("content", "")
            eq = inputs.get("Equation", {}).get("content", "")
            content = desc
            if eq:
                content += f"\n\nEquation: {eq}"
        elif node_type == "experiment":
            content = inputs.get("Description", {}).get("content", "")
            setup = inputs.get("Setup", {}).get("content", "")
            if setup:
                content += f"\n\nSetup: {setup}"
        elif node_type == "result":
            content = inputs.get("Summary", {}).get("content", "")
        elif node_type == "paper_section":
            # Include both user prompt and draft content
            prompt = inputs.get("user_prompt", {}).get("content", "")
            draft = inputs.get("draft_content", {}).get("content", "")
            content = draft if draft else prompt
        elif node_type == "finding":
            content = inputs.get("Finding", {}).get("content", "")
        elif node_type == "literature":
            content = inputs.get("Summary", {}).get("content", "")
        elif node_type == "concept":
            content = inputs.get("Definition", {}).get("content", "")
        elif node_type == "data":
            content = inputs.get("Description", {}).get("content", "")
        elif node_type == "metric":
            content = inputs.get("Definition", {}).get("content", "")
        elif node_type == "figure":
            content = inputs.get("Caption", {}).get("content", "")
        elif node_type == "table":
            content = inputs.get("Caption", {}).get("content", "")
        elif node_type == "note":
            content = inputs.get("Content", {}).get("content", "")
        else:
            # Fallback: try common field names
            for field in ["Description", "Content", "Summary", "Text"]:
                if field in inputs:
                    content = inputs.get(field, {}).get("content", "")
                    if content:
                        break
        
        # Return as dict for state (will be converted to NodeContent in compile_section_payload)
        return {
            "node_id": node["id"],
            "node_type": node_type,
            "title": data.get("title", ""),
            "content": content,
            "metadata": {
                "references": data.get("references", []),
            },
        }

    def _find_relevant_nodes(self, nodes: List[Dict], section_type: str) -> List[Dict]:
        """Find nodes relevant to a specific section type"""
        relevance_map = {
            "abstract": ["result", "finding", "method", "hypothesis"],
            "introduction": ["question", "hypothesis", "idea", "literature"],
            "related_work": ["literature", "concept"],
            "method": ["method", "experiment", "data"],
            "experiment": ["experiment", "data", "metric"],
            "result": ["result", "finding", "figure", "table"],
            "discussion": ["result", "finding", "hypothesis"],
            "conclusion": ["result", "finding", "hypothesis", "question"],
        }
        
        relevant_types = relevance_map.get(section_type, [])
        relevant_nodes = []
        
        def check_node(node):
            if node.get("type") in relevant_types:
                relevant_nodes.append(node)
            for block in node.get("blocks", []):
                check_node(block)
        
        for node in nodes:
            check_node(node)
        
        return relevant_nodes[:10]  # Limit to 10 nodes

    def _format_node_context(self, node: Dict[str, Any]) -> str:
        """Format a node dict for logging"""
        node_type = node.get("node_type", "unknown")
        title = node.get("title") or node.get("node_id", "unknown")
        content = node.get("content", "")
        parts = [f"[{node_type.upper()}] {title}"]
        if content:
            parts.append(f"  Content: {content[:500]}")
        return "\n".join(parts)

    async def run(self, 
                  work_id: str,
                  section_type: str,
                  user_prompt: str = "",
                  section_title: str = "",
                  template_id: Optional[str] = None,
                  explicit_node_ids: Optional[List[str]] = None,
                  word_count_limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the Commander Agent
        - **Description**:
            - Collects context from FlowGram.ai research graph
            - Outputs unified SectionWritePayload for Writer Agent

        - **Args**:
            - `work_id` (str): The research work ID
            - `section_type` (str): Type of section to generate
            - `user_prompt` (str): User's specific instructions
            - `section_title` (str, optional): Custom section title
            - `template_id` (str, optional): Template ID for format rules
            - `explicit_node_ids` (List[str], optional): Explicitly referenced node IDs
            - `word_count_limit` (int, optional): Target word count

        - **Returns**:
            - `dict`: Contains 'section_write_payload' (SectionWritePayload) ready for Writer
        """
        result = await self.agent.ainvoke({
            "work_id": work_id,
            "section_type": section_type,
            "section_title": section_title,
            "user_prompt": user_prompt,
            "template_id": template_id,
            "explicit_node_ids": explicit_node_ids or [],
            "word_count_limit": word_count_limit,
            "messages": [],
            "llm_calls": 0,
        })
        return result

    @property
    def name(self) -> str:
        """Agent name identifier"""
        return "commander"

    @property
    def description(self) -> str:
        """Agent description"""
        return "Orchestrates paper writing by assembling context and compiling prompts for content generation"

    @property
    def router(self) -> APIRouter:
        """Return the FastAPI router for this agent"""
        return create_commander_router(self)

    @property
    def endpoints_info(self) -> List[Dict[str, Any]]:
        """Return endpoint metadata for list_agents"""
        return [
            {
                "path": "/agent/commander/prepare",
                "method": "POST",
                "description": "Prepare context and prompt for paper section generation",
                "input_model": "CommanderPayload",
                "output_model": "CommanderResult"
            }
        ]
