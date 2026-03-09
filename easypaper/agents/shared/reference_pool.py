"""
ReferencePool - Persistent reference management across paper generation phases.

- **Description**:
    - Manages a growing pool of academic references used during paper generation.
    - Separates core references (user-provided, immutable) from discovered
      references (found via search_papers during writing).
    - Provides real-time valid_citation_keys as the pool grows.
    - Generates the final .bib file content from all accumulated references.
"""

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("uvicorn.error")


class ReferencePool:
    """
    Persistent reference pool that accumulates citations across generation phases.

    - **Description**:
        - Initialized with the user's core references (BibTeX strings).
        - During content generation, search_papers may discover new papers.
        - After validation (two-layer: LLM judgment + system cross-reference),
          new papers are added via add_discovered().
        - valid_citation_keys grows in real-time, so subsequent sections and
          mini_review always see the full, up-to-date reference set.
        - to_bibtex() produces the complete .bib content for final output.

    - **Args**:
        - `initial_bibtex_list` (List[str]): User-provided BibTeX entry strings.
    """

    def __init__(self, initial_bibtex_list: List[str]):
        self._core_refs: List[Dict[str, Any]] = self._parse_bibtex_list(
            initial_bibtex_list
        )
        self._discovered_refs: List[Dict[str, Any]] = []
        self._all_keys: Set[str] = {r["ref_id"] for r in self._core_refs if r.get("ref_id")}

    @classmethod
    async def create(
        cls,
        initial_refs: List[str],
        paper_search_config: Optional[Dict[str, Any]] = None,
    ) -> "ReferencePool":
        """
        Async factory that resolves plain-text references via search before
        falling back to heuristic conversion.

        - **Args**:
            - `initial_refs` (List[str]): User-provided reference strings
              (BibTeX or plain text).
            - `paper_search_config` (dict, optional): Config for PaperSearchTool.

        - **Returns**:
            - `ReferencePool`: Fully initialised pool with high-quality BibTeX.
        """
        from ..shared.tools.paper_search import PaperSearchTool

        cfg = paper_search_config or {}
        tool = PaperSearchTool(
            serpapi_api_key=cfg.get("serpapi_api_key"),
            semantic_scholar_api_key=cfg.get("semantic_scholar_api_key"),
            timeout=cfg.get("timeout", 10),
            semantic_scholar_min_results_before_fallback=cfg.get(
                "semantic_scholar_min_results_before_fallback", 3
            ),
            enable_query_cache=cfg.get("enable_query_cache", True),
            cache_ttl_hours=cfg.get("cache_ttl_hours", 24),
        )

        resolved: List[str] = []
        enrichment_hits: List[Dict[str, Any]] = []
        for ref_str in initial_refs:
            is_bib = bool(re.search(r"@\w+\{", ref_str))
            query = cls._extract_search_query_from_reference(ref_str)
            if not query:
                resolved.append(ref_str)
                continue
            try:
                result = await tool.execute(query=query, max_results=1)
                papers = (result.data or {}).get("papers", []) if result.success else []
                if papers:
                    top = papers[0]
                    logger.info(
                        "ref_pool.search_resolved query='%s' -> %s",
                        query[:60], top.get("bibtex_key", "?"),
                    )
                    if is_bib:
                        # Keep user-provided BibTeX key/entry for stability, enrich metadata only.
                        resolved.append(ref_str)
                        enrichment_hits.append(
                            {
                                "hint_ref_id": cls._extract_bibtex_key(ref_str),
                                "hint_title": cls._extract_bibtex_title(ref_str),
                                "paper": top,
                            }
                        )
                    elif top.get("bibtex"):
                        resolved.append(top["bibtex"])
                        # Also enrich parsed core ref with abstract/venue/citation metadata.
                        enrichment_hits.append(
                            {
                                "hint_ref_id": top.get("bibtex_key", ""),
                                "hint_title": top.get("title", ""),
                                "paper": top,
                            }
                        )
                    else:
                        resolved.append(ref_str)
                        enrichment_hits.append(
                            {
                                "hint_ref_id": "",
                                "hint_title": top.get("title", ""),
                                "paper": top,
                            }
                        )
                else:
                    logger.info("ref_pool.search_miss query='%s', using heuristic", query[:60])
                    resolved.append(ref_str)
            except Exception as exc:
                logger.warning("ref_pool.search_error query='%s': %s", query[:60], exc)
                resolved.append(ref_str)
            await asyncio.sleep(1.0)
        pool = cls(resolved)
        if enrichment_hits:
            pool._enrich_core_refs_from_search_hits(enrichment_hits)
        return pool

    @staticmethod
    def _extract_search_query(plaintext: str) -> str:
        """
        Extract a search query from a plain-text citation string.
        Tries to find the paper title; falls back to first-author + year.
        """
        plaintext = plaintext.strip().rstrip(".")
        year_match = re.search(r"\((\d{4})\)", plaintext)
        year_str = year_match.group(1) if year_match else ""

        # Use the sentence-split heuristic to find the title portion
        parts = ReferencePool._split_citation_sentences(plaintext)
        if len(parts) >= 2:
            title_candidate = parts[1].strip()
            if len(title_candidate) > 10:
                return title_candidate
        # Fallback: first author last name + year
        first_author = plaintext.split(",")[0].split("&")[0].strip()
        if first_author and year_str:
            return f"{first_author} {year_str}"
        return plaintext[:120]

    @staticmethod
    def _extract_search_query_from_reference(reference: str) -> str:
        """
        Extract a robust search query from either BibTeX or plain-text reference.
        """
        if re.search(r"@\w+\{", reference):
            title = ReferencePool._extract_bibtex_title(reference)
            if title and len(title.strip()) > 8:
                return title.strip()
            key = ReferencePool._extract_bibtex_key(reference)
            return key or reference[:120]
        return ReferencePool._extract_search_query(reference)

    @staticmethod
    def _extract_bibtex_key(bibtex: str) -> str:
        match = re.search(r"@\w+\{([^,]+),", bibtex)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _extract_bibtex_title(bibtex: str) -> str:
        match = re.search(r"title\s*=\s*[{\"]([^}\"]+)[}\"]", bibtex, re.IGNORECASE)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _norm_text(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()

    def _enrich_core_refs_from_search_hits(self, hits: List[Dict[str, Any]]) -> None:
        """
        Enrich core references with abstract/venue/citation metadata from search hits.
        """
        for hit in hits:
            paper = hit.get("paper") or {}
            if not isinstance(paper, dict):
                continue
            hint_ref_id = hit.get("hint_ref_id", "")
            hint_title = self._norm_text(hit.get("hint_title", ""))
            paper_title_norm = self._norm_text(paper.get("title", ""))
            target = None
            for core in self._core_refs:
                if hint_ref_id and core.get("ref_id") == hint_ref_id:
                    target = core
                    break
                core_title_norm = self._norm_text(core.get("title", ""))
                if hint_title and core_title_norm and core_title_norm == hint_title:
                    target = core
                    break
                if paper_title_norm and core_title_norm and core_title_norm == paper_title_norm:
                    target = core
                    break
            if not target:
                continue
            if paper.get("title"):
                target["title"] = target.get("title") or paper.get("title")
            if paper.get("year") and not target.get("year"):
                target["year"] = paper.get("year")
            if paper.get("authors"):
                existing_authors = target.get("authors")
                if not existing_authors:
                    target["authors"] = " and ".join(paper.get("authors", []))
            target["abstract"] = paper.get("abstract", target.get("abstract", ""))
            target["venue"] = paper.get("venue", target.get("venue", ""))
            target["citation_count"] = paper.get("citation_count", target.get("citation_count"))
            target["source"] = target.get("source", "core_search_enriched")

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def valid_citation_keys(self) -> Set[str]:
        """
        All currently valid citation keys (core + discovered).

        - **Returns**:
            - `Set[str]`: The complete set of valid BibTeX citation keys.
        """
        return set(self._all_keys)

    @property
    def core_refs(self) -> List[Dict[str, Any]]:
        """
        User-provided core references (read-only copy).

        - **Returns**:
            - `List[Dict]`: The core reference list.
        """
        return list(self._core_refs)

    @property
    def discovered_refs(self) -> List[Dict[str, Any]]:
        """
        References discovered during writing via search_papers (read-only copy).

        - **Returns**:
            - `List[Dict]`: The discovered reference list.
        """
        return list(self._discovered_refs)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def get_all_refs(self) -> List[Dict[str, Any]]:
        """
        Get all references (core + discovered).

        - **Returns**:
            - `List[Dict]`: Combined reference list for prompt building and
              bib generation.
        """
        return self._core_refs + self._discovered_refs

    def has_key(self, ref_id: str) -> bool:
        """
        Check if a citation key exists in the pool.

        - **Args**:
            - `ref_id` (str): The citation key to check.

        - **Returns**:
            - `bool`: True if the key is in the pool.
        """
        return ref_id in self._all_keys

    def get_ref(self, ref_id: str) -> Optional[Dict[str, Any]]:
        """
        Look up a reference by its citation key.

        - **Args**:
            - `ref_id` (str): The citation key.

        - **Returns**:
            - `Optional[Dict]`: The reference dict, or None if not found.
        """
        for ref in self._core_refs + self._discovered_refs:
            if ref.get("ref_id") == ref_id:
                return ref
        return None

    def add_discovered(
        self,
        ref_id: str,
        bibtex: str,
        source: str = "search",
    ) -> bool:
        """
        Add a validated discovered paper to the pool.

        - **Description**:
            - Skips duplicates (returns False if ref_id already exists).
            - Parses the BibTeX to extract metadata.
            - Tracks provenance via `source` field.

        - **Args**:
            - `ref_id` (str): The BibTeX citation key.
            - `bibtex` (str): The full BibTeX entry string.
            - `source` (str): Provenance label (default "search").

        - **Returns**:
            - `bool`: True if added, False if duplicate.
        """
        if ref_id in self._all_keys:
            return False
        parsed = self._parse_single_bibtex(bibtex, fallback_id=ref_id)
        parsed["source"] = source
        self._discovered_refs.append(parsed)
        self._all_keys.add(ref_id)
        return True

    def to_bibtex(self) -> str:
        """
        Generate complete .bib file content from all references.

        - **Returns**:
            - `str`: Combined BibTeX string for the entire pool.
        """
        bib_entries = []
        for ref in self.get_all_refs():
            if ref.get("bibtex"):
                bib_entries.append(ref["bibtex"])
            else:
                # Fallback: generate a minimal entry
                ref_id = ref.get("ref_id", "unknown")
                title = ref.get("title", "Unknown Title")
                authors = ref.get("authors", "Unknown Author")
                year = ref.get("year", 2024)
                entry = (
                    f"@article{{{ref_id},\n"
                    f"  title = {{{title}}},\n"
                    f"  author = {{{authors}}},\n"
                    f"  year = {{{year}}},\n"
                    f"}}"
                )
                bib_entries.append(entry)
        return "\n\n".join(bib_entries)

    def summary(self) -> str:
        """
        Human-readable summary for logging.

        - **Returns**:
            - `str`: Summary string like "core=5, discovered=3, total_keys=8".
        """
        return (
            f"core={len(self._core_refs)}, "
            f"discovered={len(self._discovered_refs)}, "
            f"total_keys={len(self._all_keys)}"
        )

    # ------------------------------------------------------------------
    # Static / class-level helpers for post-ReAct validation
    # ------------------------------------------------------------------

    @staticmethod
    def extract_cite_keys(content: str) -> Set[str]:
        """
        Extract all citation keys from LaTeX content.

        - **Description**:
            - Finds all \\cite{key1, key2, ...} patterns and returns the
              individual keys.

        - **Args**:
            - `content` (str): LaTeX content string.

        - **Returns**:
            - `Set[str]`: Set of citation keys found in the content.
        """
        keys: Set[str] = set()
        for match in re.finditer(r"\\cite\{([^}]+)\}", content):
            for key in match.group(1).split(","):
                stripped = key.strip()
                if stripped:
                    keys.add(stripped)
        return keys

    @staticmethod
    def extract_search_results_from_history(
        msg_history: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """
        Extract BibTeX entries from search_papers tool results in message history.

        - **Description**:
            - Scans messages with role='tool' for search_papers results.
            - Parses the JSON content to find bibtex_key -> bibtex mappings.

        - **Args**:
            - `msg_history` (List[Dict]): Message history from react_loop.

        - **Returns**:
            - `Dict[str, str]`: Mapping of bibtex_key to full BibTeX string.
        """
        results: Dict[str, str] = {}
        for msg in msg_history:
            if msg.get("role") != "tool":
                continue
            content_str = msg.get("content", "")
            if not content_str:
                continue
            try:
                data = json.loads(content_str)
            except (json.JSONDecodeError, TypeError):
                continue

            # ToolResult format: {"success": ..., "data": {"papers": [...], "bibtex": "..."}, ...}
            tool_data = data.get("data", {})
            if not isinstance(tool_data, dict):
                continue
            papers = tool_data.get("papers", [])
            combined_bibtex = tool_data.get("bibtex", "")

            if not papers:
                continue

            # Parse individual BibTeX entries from the combined string
            bibtex_map = ReferencePool._split_bibtex_entries(combined_bibtex)

            # Map bibtex_key from paper summaries to their BibTeX strings
            for paper in papers:
                bkey = paper.get("bibtex_key", "")
                if bkey and bkey in bibtex_map:
                    results[bkey] = bibtex_map[bkey]
                elif bkey and bkey not in results:
                    # Try to find by key in the combined string
                    entry = ReferencePool._find_bibtex_entry(combined_bibtex, bkey)
                    if entry:
                        results[bkey] = entry

        return results

    @staticmethod
    def remove_citation(content: str, key: str) -> str:
        """
        Remove a specific citation key from LaTeX content.

        - **Description**:
            - Handles single-key \\cite{key} → removes entire command.
            - Handles multi-key \\cite{a, key, b} → removes just that key.

        - **Args**:
            - `content` (str): LaTeX content.
            - `key` (str): Citation key to remove.

        - **Returns**:
            - `str`: Content with the citation key removed.
        """
        escaped_key = re.escape(key)

        # Pattern 1: sole key in \cite{key} → remove the whole \cite{}
        content = re.sub(
            rf"\\cite\{{\s*{escaped_key}\s*\}}",
            "",
            content,
        )

        # Pattern 2: key among others → remove just that key
        # \cite{a, key, b} → \cite{a, b}
        content = re.sub(
            rf",\s*{escaped_key}",
            "",
            content,
        )
        content = re.sub(
            rf"{escaped_key}\s*,\s*",
            "",
            content,
        )

        # Clean up empty cites and trailing whitespace
        content = re.sub(r"\\cite\{\s*\}", "", content)
        content = re.sub(r"  +", " ", content)
        content = re.sub(r" +([.,;:])", r"\1", content)

        return content

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_bibtex_list(self, bibtex_list: List[str]) -> List[Dict[str, Any]]:
        """
        Parse a list of reference strings into structured dicts.
        - **Description**:
            - Accepts both BibTeX entries and plain-text citations.
            - Plain-text citations are auto-converted to BibTeX format
              via heuristic parsing.

        - **Args**:
            - `bibtex_list` (List[str]): Raw reference strings (BibTeX or plain text).

        - **Returns**:
            - `List[Dict]`: Parsed reference dicts with ref_id, title,
              authors, year, bibtex fields.
        """
        parsed = []
        for ref_str in bibtex_list:
            if re.search(r"@\w+\{", ref_str):
                parsed.append(
                    self._parse_single_bibtex(ref_str, fallback_id=f"ref_{len(parsed) + 1}")
                )
            else:
                parsed.append(
                    self._convert_plaintext_to_bibtex(ref_str, index=len(parsed) + 1)
                )
        return parsed

    @staticmethod
    def _parse_single_bibtex(bibtex: str, fallback_id: str = "unknown") -> Dict[str, Any]:
        """
        Parse a single BibTeX entry string into a structured dict.

        - **Args**:
            - `bibtex` (str): Raw BibTeX string.
            - `fallback_id` (str): Fallback ref_id if parsing fails.

        - **Returns**:
            - `Dict[str, Any]`: Parsed reference dict.
        """
        try:
            ref_id_match = re.search(r"@\w+{([^,]+),", bibtex)
            title_match = re.search(
                r"title\s*=\s*[{\"]([^}\"]+)[}\"]", bibtex, re.IGNORECASE
            )
            author_match = re.search(
                r"author\s*=\s*[{\"]([^}\"]+)[}\"]", bibtex, re.IGNORECASE
            )
            year_match = re.search(
                r"year\s*=\s*[{\"]?(\d{4})[}\"]?", bibtex, re.IGNORECASE
            )

            return {
                "ref_id": ref_id_match.group(1).strip() if ref_id_match else fallback_id,
                "title": title_match.group(1) if title_match else "",
                "authors": author_match.group(1) if author_match else "",
                "year": int(year_match.group(1)) if year_match else None,
                "bibtex": bibtex,
            }
        except Exception:
            return {
                "ref_id": fallback_id,
                "bibtex": bibtex,
            }

    @staticmethod
    def _split_citation_sentences(citation: str) -> List[str]:
        """
        Split a plain-text citation into logical sentences (author / title /
        journal) without breaking on author-initial periods like "E. H.".
        """
        # Replace abbreviation periods (single capital + period) with placeholder
        protected = re.sub(r'(?<=[A-Z])\.(?=\s|,|&|\))', '\x00', citation)
        parts = [p.strip() for p in protected.split('.') if p.strip()]
        # Restore abbreviation periods
        return [p.replace('\x00', '.') for p in parts]

    @staticmethod
    def _convert_plaintext_to_bibtex(citation: str, index: int = 1) -> Dict[str, Any]:
        """
        Convert a plain-text citation string to a BibTeX entry.
        - **Description**:
            - Heuristically extracts author, year, title, and journal.
            - Uses sentence-boundary detection that skips periods after
              single capital letters (author initials).

        - **Args**:
            - `citation` (str): Plain-text citation string.
            - `index` (int): Fallback index for generating ref_id.

        - **Returns**:
            - `Dict[str, Any]`: Structured reference dict with generated BibTeX.
        """
        citation = citation.strip()
        year = None
        year_match = re.search(r'\((\d{4})\)', citation)
        if year_match:
            year = int(year_match.group(1))

        authors_str = ""
        title_str = ""
        journal_str = ""

        parts = ReferencePool._split_citation_sentences(citation)
        if len(parts) >= 3:
            authors_str = parts[0]
            title_str = parts[1]
            journal_str = parts[2]
        elif len(parts) == 2:
            authors_str = parts[0]
            title_str = parts[1]
        elif len(parts) == 1:
            title_str = parts[0]

        authors_str = re.sub(r'\s*\(\d{4}\)\s*$', '', authors_str).strip()
        journal_str = re.sub(r'\s*\d+,?\s*\w*\d*\s*\(\d{4}\)\.?$', '', journal_str).strip()

        first_author = authors_str.split(',')[0].split('&')[0].strip()
        last_name = first_author.split()[-1] if first_author else "unknown"
        last_name = re.sub(r'[^a-zA-Z]', '', last_name).lower()
        year_str = str(year) if year else "nd"
        title_words = [w.lower() for w in re.findall(r'[a-zA-Z]+', title_str) if len(w) > 3]
        title_key = title_words[0] if title_words else "ref"
        ref_id = f"{last_name}{year_str}{title_key}"

        bibtex = (
            f"@article{{{ref_id},\n"
            f"  title = {{{title_str}}},\n"
            f"  author = {{{authors_str}}},\n"
        )
        if year:
            bibtex += f"  year = {{{year}}},\n"
        if journal_str:
            bibtex += f"  journal = {{{journal_str}}},\n"
        bibtex += "}"

        return {
            "ref_id": ref_id,
            "title": title_str,
            "authors": authors_str,
            "year": year,
            "bibtex": bibtex,
        }

    @staticmethod
    def _split_bibtex_entries(combined: str) -> Dict[str, str]:
        """
        Split a combined BibTeX string into individual entries keyed by ref_id.

        - **Args**:
            - `combined` (str): Multi-entry BibTeX string.

        - **Returns**:
            - `Dict[str, str]`: Mapping from citation key to entry string.
        """
        entries: Dict[str, str] = {}
        # Split on @type{ patterns
        parts = re.split(r"(?=@\w+\{)", combined)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            key_match = re.search(r"@\w+{([^,]+),", part)
            if key_match:
                entries[key_match.group(1).strip()] = part
        return entries

    @staticmethod
    def _find_bibtex_entry(combined: str, key: str) -> Optional[str]:
        """
        Find a specific BibTeX entry by key in a combined BibTeX string.

        - **Args**:
            - `combined` (str): Multi-entry BibTeX string.
            - `key` (str): Citation key to find.

        - **Returns**:
            - `Optional[str]`: The BibTeX entry or None if not found.
        """
        entries = ReferencePool._split_bibtex_entries(combined)
        return entries.get(key)
