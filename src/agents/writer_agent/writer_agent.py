"""
Writer Agent
- **Description**:
    - Generates pure LaTeX content fragments for paper sections
    - Focuses on academic writing quality and proper citation usage
    - Supports iterative review and refinement with tool-based validation
    - Dual-mode tool invocation:
        - Type 1 (ReAct): generate_content can use react_loop with AskTool
          for consulting memory/planner/reviewer during writing
        - Type 2 (Fixed Sequence): mini_review executes CitationValidatorTool,
          WordCountTool, and KeyPointCoverageTool in fixed deterministic order
"""
from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated, Optional
from langgraph.graph import StateGraph, START, END
import operator
import re
from typing import List, Dict, Any, Set, TYPE_CHECKING
from ...config.schema import ModelConfig, ToolsConfig
from ..react_base import ReActAgent

if TYPE_CHECKING:
    from fastapi import APIRouter
from .models import GeneratedContent, ReviewResult
from ..shared.tools import (
    ToolRegistry,
    CitationValidatorTool,
    WordCountTool,
    KeyPointCoverageTool,
)


WRITER_SYSTEM_BASE = """You are an expert academic writer specializing in research paper composition.
Your task is to generate high-quality LaTeX content for a specific section of a research paper.

CRITICAL RULES:
1. Generate ONLY LaTeX body content - NO document preamble, NO \\documentclass, NO \\begin{document}
2. Use proper LaTeX formatting for sections, equations, lists, etc.
3. Use \\cite{reference_id} for all citations
4. Use \\ref{label} for cross-references
5. Use \\includegraphics{figure_id} for figures (just the ID, no path)
6. FIGURE PLACEMENT: Each figure environment (\\begin{figure}...\\end{figure}) must appear
   AT MOST ONCE in this section. NEVER create duplicate figure environments for the same figure.
   If a figure is listed as "REFERENCE ONLY", use \\ref{fig:...} to refer to it — do NOT
   create a \\begin{figure} environment for it.
7. Maintain formal academic writing style
8. Be precise and evidence-based
9. Structure content logically:
9.1 If the user prompt includes a "Structure Quality Contract", treat it as **mandatory**.
    Follow its subsection policy exactly — if it says DO NOT use \\subsection{}, you must not.
9.2 Only use \\subsection{} commands when the Structure Quality Contract explicitly recommends them.
    By default, prefer continuous narrative prose with paragraph-level transitions.
10. CITATION CONSTRAINT: You MUST ONLY use citation keys that are explicitly provided in the resources/references list. 
   DO NOT invent, hallucinate, or use any citation keys that are not in the provided list.
   DO NOT use placeholder citations like \\cite{need_citation} or similar.
   If you need to cite something but no suitable reference is provided, simply omit the \\cite command entirely and describe the concept without citation.
11. NEVER use Markdown formatting. This is a LaTeX document, NOT Markdown.
    - NO **bold** or __bold__ — use \\textbf{bold}
    - NO *italic* or _italic_ — use \\textit{italic}
    - NO ## headings — use \\subsection{} or \\subsubsection{}
    - NO - bullet lists — use \\begin{itemize} ... \\item ... \\end{itemize}
    - NO 1. numbered lists — use \\begin{enumerate} ... \\item ... \\end{enumerate}
    - NO `code` backticks — use \\texttt{code}
12. CODE EVIDENCE RULE: NEVER reference specific code file names, file paths,
    or repository structure in the generated text. Code evidence is provided
    to help you understand the methods — describe algorithms and techniques
    conceptually (e.g., "a fine-tuned BERT classifier", "convex hull volume
    computation") rather than citing implementation files (e.g., do NOT write
    "implemented in \\texttt{code/classify.py}").

ACADEMIC WRITING BASELINE (always enforce):
1. Use present tense for describing methods and general conclusions;
   use past tense only for specific experiments already conducted.
2. NO contractions: write "it is" not "it's", "do not" not "don't",
   "cannot" not "can't", "will not" not "won't".
3. NO possessives on method or model names: write "the performance of BERT"
   not "BERT's performance".
4. Subject-verb proximity: keep the grammatical subject and main verb close
   together. Do NOT insert long parenthetical clauses between them.
5. Stress position: place the most important information (the key result or
   new concept) at the END of each sentence.

FORMATTING GUIDELINES:
- For equations: use \\begin{equation} or inline $...$ 
- For lists: use \\begin{itemize} or \\begin{enumerate}
- For emphasis: use \\textit{} or \\textbf{}
- For code/algorithms: use \\begin{algorithm} or similar

OUTPUT FORMAT:
Return ONLY the LaTeX content for the section. Do not include explanations or comments outside the LaTeX."""


REVISION_SYSTEM_PROMPT = """You are revising a section of an academic paper based on review feedback.
Your task is to fix the issues identified while maintaining the overall quality and structure.

IMPORTANT:
- Fix ALL issues mentioned in the feedback
- Keep the same general structure and content
- Make minimal changes needed to address the issues
- Preserve existing valid citations and citation density unless the feedback explicitly requires removal.
- If citations were flagged as invalid, REMOVE them entirely (don't replace with other citations)
- NEVER use Markdown formatting — this is LaTeX. Use \\textbf{}, \\textit{}, \\subsection{}, \\begin{itemize}, etc.
- Preserve or improve structural clarity: thematic block boundaries, transitions, and subsection quality.
- If the original request contains a Structure Quality Contract, revision MUST continue to satisfy it.

REVISION STRATEGIES BY ISSUE TYPE:

1. EXPAND (word count too low):
   - Add concrete examples or experimental details to support claims
   - Expand terse sentences into full reasoning chains
   - Add transition sentences between paragraphs for better flow
   - Insert references to figures/tables that were not discussed
   - Do NOT pad with filler phrases or repeat existing points

2. REDUCE (word count too high):
   - Remove redundant sentences that restate the same point
   - Merge short paragraphs that cover the same sub-topic
   - Replace verbose phrases: "in order to" → "to", "due to the fact that" → "because"
   - Remove hedging phrases: "It is worth noting that X" → "X"
   - Cut the weakest supporting argument if multiple are given

3. STYLE FIX (AI-style language, contractions, etc.):
   - Replace flagged AI-style words with concrete academic alternatives
   - Expand all contractions: "it's" → "it is", "don't" → "do not"
   - Rewrite possessives on method names: "BERT's" → "the performance of BERT"
   - Break up stacked connective adverbs (Furthermore...Moreover...Additionally)
   - Move key results to sentence-final (stress) position

4. STRUCTURE FIX (block clarity / subsection quality):
   - Keep major thematic blocks distinguishable after edits
   - Add or refine transition sentences between blocks when needed
   - Do NOT add \\subsection{} commands unless the original Structure Quality Contract allows them
   - Avoid collapsing multiple themes into one undifferentiated paragraph chain

Return ONLY the revised LaTeX content."""


class WriterAgentState(TypedDict):
    """
    State for Writer Agent workflow with iterative review support.
    """
    messages: Annotated[list[AnyMessage], operator.add]
    system_prompt: Optional[str]
    user_prompt: Optional[str]
    section_type: Optional[str]
    citation_format: Optional[str]
    constraints: Optional[List[str]]
    generated_content: Optional[str]
    citation_ids: Optional[List[str]]
    figure_ids: Optional[List[str]]
    table_ids: Optional[List[str]]
    llm_calls: int
    
    # Iterative review fields
    iteration: int
    max_iterations: int
    enable_review: bool
    
    # Review context
    valid_citation_keys: Optional[List[str]]
    target_words: Optional[int]
    key_points: Optional[List[str]]
    
    # Review results
    review_result: Optional[Dict[str, Any]]
    revision_prompt: Optional[str]
    revision_plan: Optional[Dict[str, Any]]
    review_history: Annotated[List[Dict[str, Any]], operator.add]
    writer_response_section: Annotated[List[Dict[str, Any]], operator.add]
    writer_response_paragraph: Annotated[List[Dict[str, Any]], operator.add]
    
    # Final tracking
    invalid_citations_removed: Optional[List[str]]
    paragraph_units: Optional[List[Dict[str, Any]]]

    # Shared memory + peer agents for AskTool (ReAct consultation)
    memory: Optional[Any]
    peers: Optional[Dict[str, Any]]
    mode: Optional[str]
    current_content: Optional[str]


class WriterAgent(ReActAgent):
    """
    Writer Agent for generating LaTeX content with iterative review.

    - **Description**:
        - Inherits from ReActAgent for access to react_loop and setup_tools.
        - Generates academic LaTeX content based on compiled prompts.
        - Dual-mode tool invocation:
            - Type 1 (ReAct): generate_content can optionally use react_loop
              with AskTool for consulting memory/planner/reviewer.
            - Type 2 (Fixed Sequence): mini_review executes citation validation,
              word count, and key point coverage tools in fixed order.
        - Iteratively revises content based on review feedback.
        - Extracts citations and figure references from generated content.
    """

    def __init__(self, config: ModelConfig, tools_config: Optional[ToolsConfig] = None):
        if tools_config is None:
            tools_config = ToolsConfig(
                enabled=True,
                available_tools=[
                    "validate_citations",
                    "count_words",
                    "check_key_points",
                ],
                max_react_iterations=3,
            )
        super().__init__(config, tools_config)
        self.agent = self.init_agent()

    def init_agent(self):
        """
        Initialize the agent workflow graph with iterative review.
        
        Graph structure:
        START -> generate_content -> mini_review -> (decision)
            -> if passed or max_iterations: extract_references -> END
            -> if not passed: revise_content -> mini_review (loop)
        """
        agent_builder = StateGraph(WriterAgentState)
        
        # Add nodes
        agent_builder.add_node("generate_content", self.generate_content)
        agent_builder.add_node("mini_review", self.mini_review)
        agent_builder.add_node("revise_content", self.revise_content)
        agent_builder.add_node("extract_references", self.extract_references)
        
        # Add edges
        agent_builder.add_edge(START, "generate_content")
        agent_builder.add_edge("generate_content", "mini_review")
        
        # Conditional edge after mini_review
        agent_builder.add_conditional_edges(
            "mini_review",
            self._should_revise,
            {
                "revise": "revise_content",
                "done": "extract_references",
            }
        )
        
        # After revision, go back to review
        agent_builder.add_edge("revise_content", "mini_review")
        
        # Final extraction leads to end
        agent_builder.add_edge("extract_references", END)
        
        return agent_builder.compile()

    def _should_revise(self, state: WriterAgentState) -> str:
        """
        Conditional edge: decide whether to revise or finish.
        
        Returns:
            "revise" if revision needed, "done" otherwise
        """
        # Check if review is enabled
        if not state.get("enable_review", True):
            return "done"
        
        review = state.get("review_result", {})
        iteration = state.get("iteration", 1)
        max_iter = state.get("max_iterations", 2)
        
        # If review passed or max iterations reached, we're done
        if review.get("passed", True):
            print(f"[WriterAgent] Review passed at iteration {iteration}")
            return "done"
        
        if iteration >= max_iter:
            print(f"[WriterAgent] Max iterations ({max_iter}) reached")
            return "done"
        
        print(f"[WriterAgent] Revision needed (iteration {iteration}/{max_iter})")
        return "revise"

    async def generate_content(self, state: WriterAgentState) -> Dict[str, Any]:
        """
        Generate LaTeX content using ReAct loop with AskTool access.
        - **Description**:
            - Registers the ``ask`` tool (backed by memory/planner/reviewer)
              so the LLM can consult them during writing.
            - Falls back to a plain LLM call when no tools are available.
        """
        print(f"[WriterAgent] Generating content for: {state.get('section_type')}")

        system_prompt = state.get("system_prompt", "")
        user_prompt = state.get("user_prompt", "")
        mode = (state.get("mode") or "draft").strip().lower()
        citation_format = state.get("citation_format", "cite")
        memory = state.get("memory")
        peers = state.get("peers") or {}
        current_content = state.get("current_content", "") or ""
        revision_plan = state.get("revision_plan") or {}

        if mode == "revision":
            messages = [
                {"role": "system", "content": REVISION_SYSTEM_PROMPT},
                {"role": "user", "content": f"Original request:\n{user_prompt}"},
                {"role": "assistant", "content": current_content},
                {
                    "role": "user",
                    "content": (
                        "Revise the section according to the instruction and constraints.\n"
                        f"Structured revision plan:\n{revision_plan}"
                    ),
                },
            ]
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.35,
                    max_tokens=4000,
                )
                revised_content = response.choices[0].message.content or ""
                revised_content = self._clean_latex_output(revised_content)
                return {
                    "generated_content": revised_content,
                    "llm_calls": state.get("llm_calls", 0) + 1,
                    "iteration": 1,
                }
            except Exception as e:
                print(f"[WriterAgent] Error generating revision content: {e}")
                return {
                    "generated_content": current_content,
                    "llm_calls": state.get("llm_calls", 0) + 1,
                    "iteration": 1,
                }

        full_system = f"{WRITER_SYSTEM_BASE}\n\n{system_prompt}"

        if citation_format != "cite":
            full_system = full_system.replace(
                "\\cite{reference_id}",
                f"\\{citation_format}{{reference_id}}",
            )

        # Build tool context for AskTool + PaperSearchTool
        tool_context: Dict[str, Any] = {
            "valid_keys": set(state.get("valid_citation_keys", [])),
            "key_points": state.get("key_points", []),
        }
        tool_names: List[str] = []

        if memory is not None:
            tool_context["memory"] = memory
            tool_names.append("ask")
        if peers.get("planner"):
            tool_context["planner"] = peers["planner"]
            if "ask" not in tool_names:
                tool_names.append("ask")
        if peers.get("reviewer"):
            tool_context["reviewer"] = peers["reviewer"]
            if "ask" not in tool_names:
                tool_names.append("ask")

        self.setup_tools(tool_names, **tool_context)

        messages = [
            {"role": "system", "content": full_system},
            {"role": "user", "content": user_prompt},
        ]

        try:
            generated_content, _ = await self.react_loop(
                messages=messages,
                tool_names=tool_names,
                temperature=0.7,
                max_tokens=4000,
            )
            generated_content = self._clean_latex_output(generated_content)
        except Exception as e:
            print(f"[WriterAgent] Error generating content: {e}")
            generated_content = f"% Error generating content: {str(e)}"

        return {
            "generated_content": generated_content,
            "llm_calls": state.get("llm_calls", 0) + 1,
            "iteration": 1,
        }

    async def mini_review(self, state: WriterAgentState) -> Dict[str, Any]:
        """
        Perform mini-review on generated content.
        
        Checks:
        - Citation validity (using CitationValidatorTool)
        - Word count vs target (using WordCountTool)
        - Key point coverage (if key_points provided)
        """
        content = state.get("generated_content", "")
        section_type = state.get("section_type", "unknown")
        
        print(f"[WriterAgent] Mini-review for {section_type} (iteration {state.get('iteration', 1)})")
        
        issues = []
        warnings = []
        invalid_citations = []
        fixed_content = content
        
        # 1. Citation validation
        valid_keys = set(state.get("valid_citation_keys", []))
        if valid_keys:
            validator = CitationValidatorTool(valid_keys)
            result = await validator.execute(content=content, fix_invalid=True)
            
            if result.data:
                invalid_citations = result.data.get("invalid_citations", [])
                if invalid_citations:
                    issues.append(f"Invalid citations: {invalid_citations}")
                    fixed_content = result.data.get("fixed_content", content)
        
        # 2. Word count (informational only — not a pass/fail criterion)
        word_counter = WordCountTool()
        target_words = state.get("target_words")
        wc_result = await word_counter.execute(content=fixed_content, target_words=target_words)
        word_count = wc_result.data.get("word_count", 0) if wc_result.data else 0
        
        # 3. Key point coverage — all key points must be addressed
        key_points = state.get("key_points", [])
        coverage = 1.0
        missing_kps: list = []
        if key_points:
            kp_tool = KeyPointCoverageTool(key_points)
            kp_result = await kp_tool.execute(content=fixed_content)
            
            if kp_result.data:
                coverage = kp_result.data.get("coverage", 1.0)
                missing_kps = kp_result.data.get("missing", [])
                if coverage < 1.0 and missing_kps:
                    issues.append(
                        f"Missing key points ({coverage:.0%} coverage): "
                        + "; ".join(missing_kps[:5])
                    )
        
        # Determine if passed
        passed = len(issues) == 0
        
        # Build review result
        review_result = {
            "passed": passed,
            "issues": issues,
            "warnings": warnings,
            "invalid_citations": invalid_citations,
            "word_count": word_count,
            "target_words": target_words,
            "key_point_coverage": coverage,
            "missing_key_points": missing_kps,
        }
        
        # Build revision prompt if needed
        revision_prompt = None
        if not passed:
            revision_prompt = self._build_revision_prompt(review_result)
        
        # Log review results
        if issues:
            print(f"[WriterAgent] Mini-review issues: {issues}")
        if warnings:
            print(f"[WriterAgent] Mini-review warnings: {warnings}")
        
        return {
            "generated_content": fixed_content,  # Use fixed content with invalid citations removed
            "review_result": review_result,
            "revision_prompt": revision_prompt,
            "review_history": [review_result],
            "invalid_citations_removed": invalid_citations,
        }

    async def revise_content(self, state: WriterAgentState) -> Dict[str, Any]:
        """
        Revise content based on review feedback.
        """
        print(f"[WriterAgent] Revising content (iteration {state.get('iteration', 1) + 1})")
        
        revision_prompt = state.get("revision_prompt", "")
        revision_plan = state.get("revision_plan") or {}
        previous_content = state.get("generated_content", "")
        original_user_prompt = state.get("user_prompt", "")
        section_type = state.get("section_type", "unknown")
        
        # Build multi-turn conversation for revision
        messages = [
            {"role": "system", "content": REVISION_SYSTEM_PROMPT},
            {"role": "user", "content": f"Original request:\n{original_user_prompt}"},
            {"role": "assistant", "content": previous_content},
            {
                "role": "user",
                "content": (
                    f"Review feedback - please revise:\n{revision_prompt}\n\n"
                    f"Structured revision plan (must follow if provided):\n"
                    f"{revision_plan}"
                ),
            }
        ]
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.5,  # Lower temperature for revision
                max_tokens=4000,
            )
            
            revised_content = response.choices[0].message.content
            revised_content = self._clean_latex_output(revised_content)
            
        except Exception as e:
            print(f"[WriterAgent] Error revising content: {e}")
            revised_content = previous_content  # Keep previous if revision fails

        changed = revised_content.strip() != previous_content.strip()
        disposition = "executed" if changed else "no_change"
        section_target = str(
            revision_plan.get("section_type")
            or revision_plan.get("target")
            or section_type
        )
        normalized_constraints = revision_plan.get("constraints", {}) or {}
        constraints_payload = {
            "preserve_claims": (
                normalized_constraints.get("preserve_claims", [])
                if isinstance(normalized_constraints, dict)
                else revision_plan.get("preserve_claims", [])
            ),
            "do_not_change": (
                normalized_constraints.get("do_not_change", [])
                if isinstance(normalized_constraints, dict)
                else revision_plan.get("do_not_change", [])
            ),
        }
        section_response = {
            "target_id": section_target,
            "section_type": section_target,
            "instruction": str(
                revision_plan.get("instruction")
                or revision_prompt
                or "Apply upstream review instructions exactly."
            ),
            "constraints": constraints_payload,
            "disposition": disposition,
            "evidence": {
                "before_words": len(previous_content.split()),
                "after_words": len(revised_content.split()),
            },
        }
        paragraph_responses: List[Dict[str, Any]] = []
        target_paragraphs = revision_plan.get("target_paragraphs", []) or []
        para_instructions = revision_plan.get("paragraph_instructions", {}) or {}
        for pidx in target_paragraphs:
            if not isinstance(pidx, int) and not str(pidx).isdigit():
                continue
            pid = int(pidx)
            paragraph_responses.append({
                "target_id": f"{section_target}.p{pid}",
                "section_type": section_target,
                "paragraph_index": pid,
                "instruction": str(para_instructions.get(pid, para_instructions.get(str(pid), ""))),
                "constraints": constraints_payload,
                "disposition": disposition,
                "evidence": {
                    "content_changed": changed,
                },
            })
        
        return {
            "generated_content": revised_content,
            "llm_calls": state.get("llm_calls", 0) + 1,
            "iteration": state.get("iteration", 1) + 1,
            "writer_response_section": [section_response],
            "writer_response_paragraph": paragraph_responses,
        }

    def _build_revision_prompt(self, review_result: Dict[str, Any]) -> str:
        """
        Build a revision prompt from review results.
        """
        parts = ["The following issues were found in your output:"]
        
        for issue in review_result.get("issues", []):
            parts.append(f"- ISSUE: {issue}")
        
        for warning in review_result.get("warnings", []):
            parts.append(f"- WARNING: {warning}")
        
        parts.append("\nPlease revise your output to address these issues.")
        
        if review_result.get("invalid_citations"):
            parts.append(f"\nREMOVE these invalid citations completely (do not replace): {review_result['invalid_citations']}")

        # Explicit guidance for missing key points
        missing = review_result.get("missing_key_points", [])
        if missing:
            parts.append("\nYou MUST address the following key points that are currently missing:")
            for kp in missing:
                parts.append(f"  - {kp}")
        
        return "\n".join(parts)

    async def extract_references(self, state: WriterAgentState) -> Dict[str, Any]:
        """
        Extract citation and figure references from generated content,
        then persist the result in SessionMemory if available.
        """
        print(f"[WriterAgent] Extracting references")

        content = state.get("generated_content", "")
        citation_format = state.get("citation_format", "cite")
        section_type = state.get("section_type", "unknown")
        memory = state.get("memory")

        # Extract citations
        cite_pattern = rf'\\{citation_format}\{{([^}}]+)\}}'
        citation_matches = re.findall(cite_pattern, content)
        citation_ids = []
        for match in citation_matches:
            for cid in match.split(','):
                cid = cid.strip()
                if cid and cid not in citation_ids:
                    citation_ids.append(cid)

        if citation_format != "cite":
            std_matches = re.findall(r'\\cite\{([^}]+)\}', content)
            for match in std_matches:
                for cid in match.split(','):
                    cid = cid.strip()
                    if cid and cid not in citation_ids:
                        citation_ids.append(cid)

        # Extract figure references
        figure_pattern = r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}'
        figure_ids = list(set(re.findall(figure_pattern, content)))

        # Extract table references
        table_pattern = r'\\begin\{table\}.*?\\label\{([^}]+)\}'
        table_ids = list(set(re.findall(table_pattern, content, re.DOTALL)))

        # Persist to SessionMemory
        if memory is not None:
            memory.update_section(section_type, content)
            wc = len(content.split())
            memory.log(
                "writer", "generation", f"completed_{section_type}",
                narrative=f"Writer finished drafting {section_type} ({wc} words, {len(citation_ids)} citations, {state.get('iteration', 1)} iteration(s)).",
                word_count=wc,
                iterations=state.get("iteration", 1),
                citations=len(citation_ids),
            )

        paragraph_units = self._extract_paragraph_units(
            section_type=section_type,
            latex_content=content,
        )

        return {
            "citation_ids": citation_ids,
            "figure_ids": figure_ids,
            "table_ids": table_ids,
            "paragraph_units": paragraph_units,
            "writer_response_section": state.get("writer_response_section", []),
            "writer_response_paragraph": state.get("writer_response_paragraph", []),
        }

    def _extract_paragraph_units(
        self,
        section_type: str,
        latex_content: str,
    ) -> List[Dict[str, Any]]:
        """
        Extract paragraph-addressable units from section LaTeX.
        - **Description**:
            - Uses blank-line paragraph splitting for stable IDs
            - Adds a lightweight sentence split for diagnostic-level review
        """
        units: List[Dict[str, Any]] = []
        if not latex_content or not latex_content.strip():
            return units

        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", latex_content) if p.strip()]
        for idx, paragraph in enumerate(paragraphs):
            sentence_candidates = [
                s.strip()
                for s in re.split(r"(?<=[.!?])\s+", paragraph.replace("\n", " "))
                if s.strip()
            ]
            units.append(
                {
                    "paragraph_id": f"{section_type}.p{idx}",
                    "section_type": section_type,
                    "paragraph_index": idx,
                    "text": paragraph,
                    "sentence_count": len(sentence_candidates),
                    "sentences": sentence_candidates,
                }
            )
        return units

    def _clean_latex_output(self, content: str) -> str:
        """
        Clean LLM output to ensure pure LaTeX.
        - **Description**:
            - Removes markdown code fences
            - Removes accidental document structure
            - Converts residual markdown formatting to LaTeX equivalents
        """
        # Remove markdown code blocks
        content = re.sub(r'^```(?:latex|tex)?\s*\n', '', content)
        content = re.sub(r'\n```\s*$', '', content)
        content = re.sub(r'^```\s*\n', '', content)
        content = re.sub(r'\n```\s*$', '', content)
        
        # Remove any document structure if accidentally included
        content = re.sub(r'\\documentclass.*?\n', '', content)
        content = re.sub(r'\\begin\{document\}', '', content)
        content = re.sub(r'\\end\{document\}', '', content)
        content = re.sub(r'\\usepackage.*?\n', '', content)
        
        # --- Convert residual markdown formatting to LaTeX ---
        # Bold: **text** or __text__ -> \textbf{text}
        content = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', content)
        content = re.sub(r'__(.+?)__', r'\\textbf{\1}', content)
        # Italic: *text* or _text_ -> \textit{text}
        # (Be careful not to match LaTeX subscripts like a_i or already-converted \textbf{})
        content = re.sub(r'(?<![\\{])\*([^*\n]+?)\*', r'\\textit{\1}', content)
        content = re.sub(r'(?<=\s)_([^_\n]+?)_(?=[\s.,;:)])', r'\\textit{\1}', content)
        # Inline code: `text` -> \texttt{text}
        content = re.sub(r'`([^`\n]+?)`', r'\\texttt{\1}', content)
        # Headings: ## Title -> \subsection{Title}
        content = re.sub(r'^###\s+(.+)$', r'\\subsubsection{\1}', content, flags=re.MULTILINE)
        content = re.sub(r'^##\s+(.+)$', r'\\subsection{\1}', content, flags=re.MULTILINE)
        
        return content.strip()

    async def run(
        self,
        system_prompt: str,
        user_prompt: str,
        section_type: str = "introduction",
        citation_format: str = "cite",
        constraints: Optional[List[str]] = None,
        valid_citation_keys: Optional[List[str]] = None,
        target_words: Optional[int] = None,
        key_points: Optional[List[str]] = None,
        revision_plan: Optional[Dict[str, Any]] = None,
        max_iterations: int = 2,
        enable_review: bool = True,
        memory: Optional[Any] = None,
        peers: Optional[Dict[str, Any]] = None,
        mode: str = "draft",
        current_content: Optional[str] = None,
    ):
        """
        Run the Writer Agent with iterative review and ReAct consultation.

        - **Args**:
            - `system_prompt` (str): Full system prompt with context
            - `user_prompt` (str): Writing instruction
            - `section_type` (str): Type of section being written
            - `citation_format` (str): Citation command format
            - `constraints` (List[str], optional): Additional constraints
            - `valid_citation_keys` (List[str], optional): Valid citation keys
            - `target_words` (int, optional): Target word count
            - `key_points` (List[str], optional): Key points to cover
            - `revision_plan` (Dict[str, Any], optional): Paragraph-level revision constraints
            - `max_iterations` (int): Maximum revision iterations
            - `enable_review` (bool): Whether to enable mini-review
            - `memory` (SessionMemory, optional): Shared session memory
            - `peers` (Dict, optional): Peer agents for AskTool routing
              e.g. ``{"planner": planner_agent, "reviewer": reviewer_agent}``

        - **Returns**:
            - `dict`: Generated content and review results
        """
        initial_state = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "section_type": section_type,
            "citation_format": citation_format,
            "constraints": constraints or [],
            "messages": [],
            "llm_calls": 0,
            # Review context
            "valid_citation_keys": valid_citation_keys or [],
            "target_words": target_words,
            "key_points": key_points or [],
            "revision_plan": revision_plan or {},
            "max_iterations": max_iterations,
            "enable_review": enable_review,
            # Initialize iteration
            "iteration": 0,
            "review_history": [],
            "writer_response_section": [],
            "writer_response_paragraph": [],
            # Shared memory + peers for ReAct AskTool
            "memory": memory,
            "peers": peers,
            "paragraph_units": [],
            "mode": mode,
            "current_content": current_content,
        }

        return await self.agent.ainvoke(initial_state)

    @property
    def name(self) -> str:
        return "writer"

    @property
    def description(self) -> str:
        return "Generates LaTeX content with iterative review for academic quality"

    @property
    def router(self) -> "APIRouter | None":
        try:
            from .router import create_writer_router
            return create_writer_router(self)
        except Exception:
            return None

    @property
    def endpoints_info(self) -> List[Dict[str, Any]]:
        return [
            {
                "path": "/agent/writer/generate",
                "method": "POST",
                "description": "Generate LaTeX content for a paper section with iterative review",
                "input_model": "WriterPayload",
                "output_model": "WriterResult"
            },
            {
                "path": "/agent/writer/write-section",
                "method": "POST",
                "description": "Direct section writing with provided context (standalone API)",
                "input_model": "SectionWritePayload",
                "output_model": "SectionWriteResult"
            }
        ]
