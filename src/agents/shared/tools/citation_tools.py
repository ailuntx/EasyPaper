"""
Citation and word count tools for Writer Agent.

These tools provide validation and counting capabilities during
content generation and review.
"""

import re
from typing import Any, Dict, List, Set, Optional
from .base import WriterTool, ToolResult


class CitationValidatorTool(WriterTool):
    """
    Tool to validate citations in LaTeX content.
    
    Checks that all \\cite{} commands reference valid citation keys
    and can optionally fix invalid citations.
    """
    
    def __init__(self, valid_keys: Set[str]):
        """
        Initialize with the set of valid citation keys.
        
        Args:
            valid_keys: Set of valid BibTeX citation keys
        """
        self._valid_keys = valid_keys
    
    @property
    def name(self) -> str:
        return "validate_citations"
    
    @property
    def description(self) -> str:
        return (
            "Validates that all \\cite{} commands in LaTeX content "
            "reference valid citation keys from the provided bibliography. "
            "Returns list of invalid citations found."
        )
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "LaTeX content to validate"
                },
                "fix_invalid": {
                    "type": "boolean",
                    "description": "Whether to return fixed content with invalid citations removed",
                    "default": False
                }
            },
            "required": ["content"]
        }
    
    async def execute(
        self, 
        content: str,
        fix_invalid: bool = False,
        **kwargs
    ) -> ToolResult:
        """
        Validate citations in the content.
        
        Args:
            content: LaTeX content to validate
            fix_invalid: Whether to fix invalid citations
            
        Returns:
            ToolResult with validation results
        """
        print(f"[Tool:validate_citations] Checking citations against {len(self._valid_keys)} valid keys...")
        
        # Find all citation keys in the content
        cite_pattern = r'\\cite\{([^}]+)\}'
        
        invalid_keys: List[str] = []
        valid_keys_used: List[str] = []
        
        def process_cite(match):
            cite_content = match.group(1)
            keys = [k.strip() for k in cite_content.split(',')]
            
            valid_in_cite = []
            for key in keys:
                if key in self._valid_keys:
                    valid_in_cite.append(key)
                    if key not in valid_keys_used:
                        valid_keys_used.append(key)
                else:
                    if key not in invalid_keys:
                        invalid_keys.append(key)
            
            if fix_invalid:
                if valid_in_cite:
                    return f'\\cite{{{", ".join(valid_in_cite)}}}'
                else:
                    return ''
            else:
                return match.group(0)
        
        fixed_content = re.sub(cite_pattern, process_cite, content)
        
        if fix_invalid:
            # Clean up empty citations and dangling text
            fixed_content = re.sub(r'\\cite\{\s*\}', '', fixed_content)
            fixed_content = re.sub(r'  +', ' ', fixed_content)
            fixed_content = re.sub(r' +([.,;:])', r'\1', fixed_content)
        
        data = {
            "valid_citations": valid_keys_used,
            "invalid_citations": invalid_keys,
            "total_valid": len(valid_keys_used),
            "total_invalid": len(invalid_keys),
            "all_valid": len(invalid_keys) == 0,
        }
        
        if fix_invalid:
            data["fixed_content"] = fixed_content
        
        if invalid_keys:
            message = f"Found {len(invalid_keys)} invalid citation(s): {invalid_keys}"
            print(f"[Tool:validate_citations] INVALID: {invalid_keys}")
        else:
            message = f"All {len(valid_keys_used)} citations are valid"
            print(f"[Tool:validate_citations] OK: {len(valid_keys_used)} valid citations")
        
        return ToolResult(
            success=True,
            data=data,
            message=message,
        )
    
    def update_valid_keys(self, new_keys: Set[str]) -> None:
        """
        Update the set of valid citation keys.
        
        Args:
            new_keys: New set of valid keys
        """
        self._valid_keys = new_keys
    
    def add_valid_key(self, key: str) -> None:
        """
        Add a single valid citation key.
        
        Args:
            key: The citation key to add
        """
        self._valid_keys.add(key)


class WordCountTool(WriterTool):
    """
    Tool to count words in LaTeX content.
    
    Strips LaTeX commands and counts actual text words.
    """
    
    @property
    def name(self) -> str:
        return "count_words"
    
    @property
    def description(self) -> str:
        return (
            "Counts the number of words in LaTeX content, "
            "excluding LaTeX commands and environments. "
            "Useful for checking against word limits."
        )
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "LaTeX content to count words in"
                },
                "target_words": {
                    "type": "integer",
                    "description": "Optional target word count for comparison"
                }
            },
            "required": ["content"]
        }
    
    async def execute(
        self,
        content: str,
        target_words: Optional[int] = None,
        **kwargs
    ) -> ToolResult:
        """
        Count words in the content.
        
        Args:
            content: LaTeX content to count
            target_words: Optional target for comparison
            
        Returns:
            ToolResult with word count data
        """
        print(f"[Tool:count_words] Counting words (target: {target_words or 'none'})...")
        
        # Strip LaTeX commands for more accurate word count
        stripped = self._strip_latex(content)
        word_count = len(stripped.split())
        
        data = {
            "word_count": word_count,
            "character_count": len(stripped),
        }
        
        message = f"Word count: {word_count}"
        
        if target_words:
            difference = word_count - target_words
            percentage = (word_count / target_words * 100) if target_words > 0 else 0
            
            data["target_words"] = target_words
            data["difference"] = difference
            data["percentage"] = round(percentage, 1)
            
            if abs(difference) <= target_words * 0.1:
                data["status"] = "on_target"
                message += f" (on target: {target_words})"
                print(f"[Tool:count_words] OK: {word_count} words (target: {target_words}, {percentage:.0f}%)")
            elif difference > 0:
                data["status"] = "over_limit"
                message += f" ({difference} words over target {target_words})"
                print(f"[Tool:count_words] OVER: {word_count} words (+{difference} over {target_words})")
            else:
                data["status"] = "under_limit"
                message += f" ({abs(difference)} words under target {target_words})"
                print(f"[Tool:count_words] UNDER: {word_count} words ({abs(difference)} under {target_words})")
        else:
            print(f"[Tool:count_words] Result: {word_count} words")
        
        return ToolResult(
            success=True,
            data=data,
            message=message,
        )
    
    def _strip_latex(self, content: str) -> str:
        """
        Strip LaTeX commands from content for word counting.
        
        Args:
            content: Raw LaTeX content
            
        Returns:
            Content with LaTeX commands removed
        """
        # Remove comments
        content = re.sub(r'%.*$', '', content, flags=re.MULTILINE)
        
        # Remove common environments that don't contain prose
        content = re.sub(r'\\begin\{(equation|align|figure|table|algorithm)\*?\}.*?\\end\{\1\*?\}', 
                         '', content, flags=re.DOTALL)
        
        # Remove inline math
        content = re.sub(r'\$[^$]+\$', 'MATH', content)
        
        # Remove display math
        content = re.sub(r'\\\[.*?\\\]', 'MATH', content, flags=re.DOTALL)
        
        # Remove common LaTeX commands but keep their arguments
        content = re.sub(r'\\(textbf|textit|emph|cite|ref|label)\{([^}]*)\}', r'\2', content)
        
        # Remove commands without arguments
        content = re.sub(r'\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})?', '', content)
        
        # Remove remaining braces
        content = re.sub(r'[{}]', '', content)
        
        # Clean up whitespace
        content = re.sub(r'\s+', ' ', content)
        
        return content.strip()


class KeyPointCoverageTool(WriterTool):
    """
    Tool to check coverage of key points in content.
    
    Performs basic keyword matching to see if key points
    from the plan are addressed in the content.
    """
    
    def __init__(self, key_points: List[str] = None):
        """
        Initialize with optional key points.
        
        Args:
            key_points: List of key points to check for
        """
        self._key_points = key_points or []
    
    @property
    def name(self) -> str:
        return "check_key_points"
    
    @property
    def description(self) -> str:
        return (
            "Checks if key points from the plan are covered in the content. "
            "Uses keyword matching to determine coverage."
        )
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "LaTeX content to check"
                },
                "key_points": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of key points to check for"
                }
            },
            "required": ["content"]
        }
    
    async def execute(
        self,
        content: str,
        key_points: List[str] = None,
        **kwargs
    ) -> ToolResult:
        """
        Check key point coverage.
        
        Args:
            content: LaTeX content to check
            key_points: Optional override of key points
            
        Returns:
            ToolResult with coverage data
        """
        points = key_points or self._key_points
        
        print(f"[Tool:check_key_points] Checking coverage of {len(points)} key points...")
        
        if not points:
            print(f"[Tool:check_key_points] No key points to check")
            return ToolResult(
                success=True,
                data={"coverage": 1.0, "covered": [], "missing": []},
                message="No key points to check"
            )
        
        content_lower = content.lower()
        covered = []
        missing = []
        
        for point in points:
            # Extract key words from the point (words longer than 4 chars)
            key_words = [w for w in point.lower().split() if len(w) > 4][:3]
            
            if any(kw in content_lower for kw in key_words):
                covered.append(point)
            else:
                missing.append(point)
        
        coverage = len(covered) / len(points) if points else 1.0
        
        data = {
            "coverage": round(coverage, 2),
            "covered_count": len(covered),
            "total_count": len(points),
            "covered": covered,
            "missing": missing,
        }
        
        if coverage >= 0.8:
            message = f"Good coverage: {len(covered)}/{len(points)} key points addressed"
            print(f"[Tool:check_key_points] GOOD: {len(covered)}/{len(points)} ({coverage:.0%})")
        elif coverage >= 0.5:
            message = f"Partial coverage: {len(covered)}/{len(points)} key points addressed"
            print(f"[Tool:check_key_points] PARTIAL: {len(covered)}/{len(points)} ({coverage:.0%})")
        else:
            message = f"Low coverage: {len(covered)}/{len(points)} key points addressed"
            print(f"[Tool:check_key_points] LOW: {len(covered)}/{len(points)} ({coverage:.0%})")
        
        return ToolResult(
            success=True,
            data=data,
            message=message,
        )
    
    def set_key_points(self, points: List[str]) -> None:
        """Update the key points to check for."""
        self._key_points = points
