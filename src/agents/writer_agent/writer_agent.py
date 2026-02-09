"""
Writer Agent
- **Description**:
    - Generates pure LaTeX content fragments for paper sections
    - Focuses on academic writing quality and proper citation usage
    - Supports iterative review and refinement with tool-based validation
"""
from openai import AsyncOpenAI
from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated, Optional
from langgraph.graph import StateGraph, START, END
import operator
import re
from typing import List, Dict, Any, Set
from fastapi import APIRouter
from ...config.schema import ModelConfig
from ..base import BaseAgent
from .router import create_writer_router
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
6. Maintain formal academic writing style
7. Be precise and evidence-based
8. Structure content logically with appropriate subsections if needed
9. CITATION CONSTRAINT: You MUST ONLY use citation keys that are explicitly provided in the resources/references list. 
   DO NOT invent, hallucinate, or use any citation keys that are not in the provided list.
   DO NOT use placeholder citations like \\cite{need_citation} or similar.
   If you need to cite something but no suitable reference is provided, simply omit the \\cite command entirely and describe the concept without citation.
10. NEVER use Markdown formatting. This is a LaTeX document, NOT Markdown.
    - NO **bold** or __bold__ — use \\textbf{bold}
    - NO *italic* or _italic_ — use \\textit{italic}
    - NO ## headings — use \\subsection{} or \\subsubsection{}
    - NO - bullet lists — use \\begin{itemize} ... \\item ... \\end{itemize}
    - NO 1. numbered lists — use \\begin{enumerate} ... \\item ... \\end{enumerate}
    - NO `code` backticks — use \\texttt{code}

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
- If citations were flagged as invalid, REMOVE them entirely (don't replace with other citations)
- NEVER use Markdown formatting — this is LaTeX. Use \\textbf{}, \\textit{}, \\subsection{}, \\begin{itemize}, etc.

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
    review_history: Annotated[List[Dict[str, Any]], operator.add]
    
    # Final tracking
    invalid_citations_removed: Optional[List[str]]


class WriterAgent(BaseAgent):
    """
    Writer Agent for generating LaTeX content with iterative review.
    
    - **Description**:
        - Generates academic LaTeX content based on compiled prompts
        - Performs mini-review after generation to validate citations and word count
        - Iteratively revises content based on review feedback
        - Extracts citations and figure references from generated content
    """
    
    def __init__(self, config: ModelConfig):
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )
        self.model_name = config.model_name
        self.tool_registry = ToolRegistry()
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
        Generate LaTeX content using LLM.
        """
        print(f"[WriterAgent] Generating content for: {state.get('section_type')}")
        
        system_prompt = state.get("system_prompt", "")
        user_prompt = state.get("user_prompt", "")
        citation_format = state.get("citation_format", "cite")
        
        # Build the full system prompt
        full_system = f"{WRITER_SYSTEM_BASE}\n\n{system_prompt}"
        
        # Adjust citation format instruction if needed
        if citation_format != "cite":
            full_system = full_system.replace("\\cite{reference_id}", f"\\{citation_format}{{reference_id}}")
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": full_system},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=4000,
            )
            
            generated_content = response.choices[0].message.content
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
        
        # 2. Word count check
        word_counter = WordCountTool()
        target_words = state.get("target_words")
        wc_result = await word_counter.execute(content=fixed_content, target_words=target_words)
        
        word_count = wc_result.data.get("word_count", 0) if wc_result.data else 0
        
        if target_words:
            min_words = int(target_words * 0.7)
            max_words = int(target_words * 1.3)
            
            if word_count < min_words:
                issues.append(f"Word count too low: {word_count} < {min_words} (target: {target_words})")
            elif word_count > max_words:
                warnings.append(f"Word count high: {word_count} > {max_words} (target: {target_words})")
        
        # 3. Key point coverage
        key_points = state.get("key_points", [])
        coverage = 1.0
        if key_points:
            kp_tool = KeyPointCoverageTool(key_points)
            kp_result = await kp_tool.execute(content=fixed_content)
            
            if kp_result.data:
                coverage = kp_result.data.get("coverage", 1.0)
                if coverage < 0.5:
                    missing = kp_result.data.get("missing", [])
                    warnings.append(f"Low key point coverage ({coverage:.0%}): {missing[:3]}")
        
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
        previous_content = state.get("generated_content", "")
        original_user_prompt = state.get("user_prompt", "")
        
        # Build multi-turn conversation for revision
        messages = [
            {"role": "system", "content": REVISION_SYSTEM_PROMPT},
            {"role": "user", "content": f"Original request:\n{original_user_prompt}"},
            {"role": "assistant", "content": previous_content},
            {"role": "user", "content": f"Review feedback - please revise:\n{revision_prompt}"}
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
        
        return {
            "generated_content": revised_content,
            "llm_calls": state.get("llm_calls", 0) + 1,
            "iteration": state.get("iteration", 1) + 1,
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
        
        if review_result.get("word_count") and review_result.get("target_words"):
            wc = review_result["word_count"]
            target = review_result["target_words"]
            if wc < target * 0.7:
                parts.append(f"\nExpand content to approximately {target} words (currently {wc}).")
            elif wc > target * 1.3:
                parts.append(f"\nReduce content to approximately {target} words (currently {wc}).")
        
        return "\n".join(parts)

    async def extract_references(self, state: WriterAgentState) -> Dict[str, Any]:
        """
        Extract citation and figure references from generated content.
        """
        print(f"[WriterAgent] Extracting references")
        
        content = state.get("generated_content", "")
        citation_format = state.get("citation_format", "cite")
        
        # Extract citations
        cite_pattern = rf'\\{citation_format}\{{([^}}]+)\}}'
        citation_matches = re.findall(cite_pattern, content)
        citation_ids = []
        for match in citation_matches:
            for cid in match.split(','):
                cid = cid.strip()
                if cid and cid not in citation_ids:
                    citation_ids.append(cid)
        
        # Also check for standard \cite if format is different
        if citation_format != "cite":
            std_matches = re.findall(r'\\cite\{([^}]+)\}', content)
            for match in std_matches:
                for cid in match.split(','):
                    cid = cid.strip()
                    if cid and cid not in citation_ids:
                        citation_ids.append(cid)
        
        # Extract figure references
        figure_pattern = r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}'
        figure_ids = re.findall(figure_pattern, content)
        figure_ids = list(set(figure_ids))
        
        # Extract table references
        table_pattern = r'\\begin\{table\}.*?\\label\{([^}]+)\}'
        table_ids = re.findall(table_pattern, content, re.DOTALL)
        table_ids = list(set(table_ids))
        
        return {
            "citation_ids": citation_ids,
            "figure_ids": figure_ids,
            "table_ids": table_ids,
        }

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

    async def run(self,
                  system_prompt: str,
                  user_prompt: str,
                  section_type: str = "introduction",
                  citation_format: str = "cite",
                  constraints: Optional[List[str]] = None,
                  valid_citation_keys: Optional[List[str]] = None,
                  target_words: Optional[int] = None,
                  key_points: Optional[List[str]] = None,
                  max_iterations: int = 2,
                  enable_review: bool = True):
        """
        Run the Writer Agent with iterative review.
        
        Args:
            system_prompt: Full system prompt with context
            user_prompt: User's writing instruction
            section_type: Type of section being written
            citation_format: Citation command format
            constraints: Additional constraints
            valid_citation_keys: List of valid citation keys for validation
            target_words: Target word count for the section
            key_points: Key points that should be covered
            max_iterations: Maximum revision iterations (default 2)
            enable_review: Whether to enable mini-review (default True)

        Returns:
            dict: Generated content and review results
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
            "max_iterations": max_iterations,
            "enable_review": enable_review,
            # Initialize iteration
            "iteration": 0,
            "review_history": [],
        }
        
        return await self.agent.ainvoke(initial_state)

    @property
    def name(self) -> str:
        return "writer"

    @property
    def description(self) -> str:
        return "Generates LaTeX content with iterative review for academic quality"

    @property
    def router(self) -> APIRouter:
        return create_writer_router(self)

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
