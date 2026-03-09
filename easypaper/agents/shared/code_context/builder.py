"""
Code repository ingestion and writing-driven context builder.
"""

from __future__ import annotations

import fnmatch
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...metadata_agent.models import CodeRepositorySpec


DEFAULT_INCLUDE_GLOBS = [
    "**/*.py",
    "**/*.ipynb",
    "**/*.jl",
    "**/*.c",
    "**/*.cc",
    "**/*.cpp",
    "**/*.h",
    "**/*.hpp",
    "**/*.md",
    "**/*.markdown",
    "**/*.yaml",
    "**/*.yml",
    "**/*.json",
    "**/*.toml",
    "**/*.r",
    "**/*.R",
]

DEFAULT_EXCLUDE_GLOBS = [
    "**/.git/**",
    "**/node_modules/**",
    "**/venv/**",
    "**/.venv/**",
    "**/__pycache__/**",
    "**/build/**",
    "**/dist/**",
    "**/.idea/**",
    "**/.vscode/**",
]

METHOD_KEYWORDS = (
    "model", "algorithm", "architecture", "module", "class ", "def ",
    "forward", "inference", "encode", "decode", "optimizer", "pipeline",
)
EXPERIMENT_KEYWORDS = (
    "train", "eval", "experiment", "ablation", "metric", "dataset",
    "benchmark", "config", "seed", "reproduce", "hyperparameter", "validation",
)
RESULT_KEYWORDS = (
    "result", "analysis", "plot", "table", "figure", "report",
    "compare", "improvement", "error", "accuracy", "f1", "auc",
)
CONFIG_HINTS = ("config", "yaml", "yml", "json", "toml", "args", "setting")
SCRIPT_HINTS = ("train", "eval", "run", "experiment", "benchmark", "test")


@dataclass
class FileSummary:
    """
    Summarized file record used for writing-oriented evidence retrieval.
    """
    path: str
    extension: str
    size: int
    symbols: List[str]
    summary: str
    snippet: str
    lower_text: str
    method_score: int
    experiment_score: int
    result_score: int
    role_tags: List[str]
    evidence_strength: int


def _score_by_keywords(text: str, keywords: Tuple[str, ...]) -> int:
    return sum(text.count(k) for k in keywords)


def _extract_symbols(text: str, ext: str, max_items: int = 12) -> List[str]:
    symbols: List[str] = []
    if ext in {".py", ".r"}:
        symbols.extend(re.findall(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", text, flags=re.M))
        symbols.extend(re.findall(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\s*[\(:]", text, flags=re.M))
        symbols.extend(re.findall(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*<-\s*function\s*\(", text, flags=re.M))
    elif ext in {".c", ".cc", ".cpp", ".h", ".hpp"}:
        symbols.extend(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\([^;{}]*\)\s*\{", text))
    elif ext in {".md", ".markdown"}:
        symbols.extend(re.findall(r"^\s*#+\s+(.+)$", text, flags=re.M))
    return symbols[:max_items]


def _infer_role_tags(path: str, lower_text: str, ext: str) -> List[str]:
    text = f"{path.lower()} {lower_text[:2000]}"
    tags: List[str] = []
    if any(h in text for h in CONFIG_HINTS) or ext in {".yaml", ".yml", ".json", ".toml"}:
        tags.append("config")
    if any(h in text for h in SCRIPT_HINTS):
        tags.append("experiment_script")
    if "metric" in text or "accuracy" in text or "f1" in text or "auc" in text:
        tags.append("metric_logic")
    if "dataset" in text or "dataloader" in text or "preprocess" in text:
        tags.append("data_pipeline")
    if "model" in text or "forward" in text or "architecture" in text:
        tags.append("core_method")
    if "result" in text or "analysis" in text or "plot" in text:
        tags.append("result_analysis")
    if "limitation" in text or "todo" in text or "warning" in text:
        tags.append("risk_signal")
    base_name = os.path.basename(path).lower()
    if base_name in {"readme.md", "readme.markdown", "readme"}:
        tags.append("project_overview")
    return list(dict.fromkeys(tags))


def _readme_priority_boost(path: str) -> int:
    """
    Give README-like files a strong ranking boost for writing context.
    """
    name = os.path.basename(path).lower()
    if name in {"readme.md", "readme.markdown", "readme"}:
        return 12
    if "readme" in name:
        return 6
    return 0


def _build_summary(path: str, symbols: List[str], text: str, role_tags: List[str]) -> str:
    """
    Build a semantic summary WITHOUT exposing the raw file path.
    - **Description**:
        - Produces a human-readable description of what the file contains
        - Uses symbol names, role tags, or first line of content
        - File paths are deliberately excluded to prevent leakage into
          generated paper text.
    """
    # Use just the stem as a short module hint (no directory or extension)
    stem = os.path.splitext(os.path.basename(path))[0]
    first_non_empty = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            first_non_empty = stripped
            break
    if symbols:
        return f"[{stem}] defines: {', '.join(symbols[:4])}"
    if role_tags:
        return f"[{stem}] role: {'/'.join(role_tags[:2])} implementation"
    if first_non_empty:
        return f"[{stem}] {first_non_empty[:120]}"
    return f"[{stem}] supporting code module"


def _safe_read_text(path: Path, max_bytes: int = 256_000) -> Optional[str]:
    try:
        raw = path.read_bytes()
    except Exception:
        return None
    if b"\x00" in raw[:4096]:
        return None
    if len(raw) > max_bytes:
        raw = raw[:max_bytes]
    return raw.decode("utf-8", errors="ignore")


def _apply_glob_filters(rel_path: str, include_globs: List[str], exclude_globs: List[str]) -> bool:
    normalized = rel_path.replace("\\", "/")
    if include_globs and not any(fnmatch.fnmatch(normalized, g) for g in include_globs):
        return False
    if exclude_globs and any(fnmatch.fnmatch(normalized, g) for g in exclude_globs):
        return False
    return True


def _pick_top(files: List[FileSummary], attr: str, top_k: int = 8) -> List[FileSummary]:
    ranked = sorted(files, key=lambda x: (getattr(x, attr), x.evidence_strength, x.size), reverse=True)
    return [f for f in ranked if getattr(f, attr) > 0][:top_k]


def _evidence_confidence(score: int) -> float:
    return round(min(0.95, 0.45 + 0.06 * max(score, 1)), 2)


def _section_alias(section_type: str) -> str:
    mapping = {
        "method": "method",
        "methods": "method",
        "experiment": "experiment",
        "experiments": "experiment",
        "result": "result",
        "results": "result",
        "introduction": "introduction",
        "related_work": "introduction",
        "discussion": "discussion",
        "conclusion": "discussion",
        "abstract": "introduction",
    }
    return mapping.get(section_type, "method")


class CodeContextBuilder:
    """
    Build writing-oriented context assets from an optional code repository.
    """

    def __init__(self, workspace_root: Optional[str] = None):
        self.workspace_root = Path(workspace_root or os.getcwd())

    async def build(self, code_repo: CodeRepositorySpec, paper_title: str = "") -> Dict[str, Any]:
        source_path, cleanup_dir = self._resolve_source(code_repo)
        try:
            scoped_path = source_path
            if code_repo.subdir:
                scoped_path = (source_path / code_repo.subdir).resolve()
                if not scoped_path.exists() or not scoped_path.is_dir():
                    raise ValueError(f"code_repository.subdir not found: {code_repo.subdir}")

            include_globs = code_repo.include_globs or list(DEFAULT_INCLUDE_GLOBS)
            exclude_globs = list(DEFAULT_EXCLUDE_GLOBS)
            if code_repo.exclude_globs:
                exclude_globs.extend(code_repo.exclude_globs)

            files, stats = self._scan_and_summarize(
                root=scoped_path,
                include_globs=include_globs,
                exclude_globs=exclude_globs,
                max_files=code_repo.max_files,
                max_total_bytes=code_repo.max_total_bytes,
            )

            method_files = _pick_top(files, "method_score")
            experiment_files = _pick_top(files, "experiment_score")
            result_files = _pick_top(files, "result_score")
            evidence_graph = self._build_code_evidence_graph(files)
            writing_assets = self._build_writing_assets(method_files, experiment_files, result_files)
            section_asset_packs = self._build_section_asset_packs(evidence_graph, writing_assets)
            claim_support_candidates = self._build_claim_support_candidates(evidence_graph)

            return {
                "repository_info": {
                    "type": code_repo.type.value,
                    "source": str(scoped_path),
                    "ref": code_repo.ref,
                    "paper_title": paper_title,
                },
                "scan_stats": stats,
                "repo_overview": [f.summary for f in files[:16]],
                # Legacy fields (kept for backward compatibility)
                "method_pack": self._to_evidence_pack(method_files, "method"),
                "experiment_pack": self._to_evidence_pack(experiment_files, "experiment"),
                "result_pack": self._to_evidence_pack(result_files, "result"),
                # New writing-driven fields
                "code_evidence_graph": evidence_graph,
                "writing_assets": writing_assets,
                "section_asset_packs": section_asset_packs,
                "claim_support_candidates": claim_support_candidates,
                "index": [
                    {
                        "path": f.path,
                        "summary": f.summary,
                        "symbols": f.symbols,
                        "snippet": f.snippet,
                        "lower_text": f.lower_text,
                        "role_tags": f.role_tags,
                    }
                    for f in files
                ],
            }
        finally:
            if cleanup_dir and cleanup_dir.exists():
                shutil.rmtree(cleanup_dir, ignore_errors=True)

    def retrieve_for_section(
        self,
        context: Dict[str, Any],
        section_type: str,
        query_bundle: List[str],
        top_k: int = 4,
    ) -> List[Dict[str, Any]]:
        if not context or not context.get("index"):
            return []
        alias = _section_alias(section_type)
        queries = [q.strip().lower() for q in query_bundle if q and q.strip()]
        selected: List[Dict[str, Any]] = []
        seen_paths = set()

        section_pack = (context.get("section_asset_packs", {}) or {}).get(alias, {})
        evidence_by_id = {
            e.get("evidence_id", ""): e
            for e in (context.get("code_evidence_graph", []) or [])
            if e.get("evidence_id")
        }
        for ev_id in section_pack.get("evidence_ids", [])[:top_k]:
            ev = evidence_by_id.get(ev_id)
            if not ev:
                continue
            path = ev.get("path", "")
            if not path or path in seen_paths:
                continue
            selected.append(
                {
                    "evidence_id": ev_id,
                    "path": path,
                    "symbol": ", ".join((ev.get("symbols") or [])[:3]),
                    "snippet": ev.get("snippet", ""),
                    "why_relevant": ev.get("purpose", f"Supports {alias} writing"),
                    "confidence": ev.get("confidence", 0.7),
                }
            )
            seen_paths.add(path)

        ranked: List[Tuple[int, Dict[str, Any]]] = []
        for item in context["index"]:
            path = item.get("path", "")
            if path in seen_paths:
                continue
            text = f"{item.get('summary', '')}\n{item.get('lower_text', '')}"
            score = 0
            for q in queries:
                score += text.count(q)
            if alias == "method":
                score += text.count("method") + text.count("algorithm")
            elif alias == "experiment":
                score += text.count("experiment") + text.count("dataset") + text.count("metric")
            elif alias == "result":
                score += text.count("result") + text.count("analysis")
            if score > 0:
                ranked.append((score, item))

        ranked.sort(key=lambda x: x[0], reverse=True)
        for score, item in ranked:
            if len(selected) >= top_k:
                break
            selected.append(
                {
                    "path": item.get("path", ""),
                    "symbol": ", ".join(item.get("symbols", [])[:3]),
                    "snippet": item.get("snippet", ""),
                    "why_relevant": f"Matched runtime writing query for {alias} (score={score})",
                    "confidence": _evidence_confidence(score),
                }
            )

        return selected

    def _resolve_source(self, code_repo: CodeRepositorySpec) -> Tuple[Path, Optional[Path]]:
        if code_repo.type.value == "local_dir":
            source = Path(code_repo.path or "").expanduser()
            if not source.is_absolute():
                source = (self.workspace_root / source).resolve()
            if not source.exists() or not source.is_dir():
                raise FileNotFoundError(f"Local code repository path not found: {source}")
            return source, None

        temp_dir = Path(tempfile.mkdtemp(prefix="easy_paper_repo_"))
        clone_cmd = ["git", "clone", "--depth", "1"]
        if code_repo.ref:
            clone_cmd.extend(["--branch", code_repo.ref])
        clone_cmd.extend([code_repo.url or "", str(temp_dir)])
        result = subprocess.run(clone_cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise RuntimeError(f"Failed to clone git repo: {stderr}")
        return temp_dir, temp_dir

    def _scan_and_summarize(
        self,
        root: Path,
        include_globs: List[str],
        exclude_globs: List[str],
        max_files: int,
        max_total_bytes: int,
    ) -> Tuple[List[FileSummary], Dict[str, Any]]:
        files: List[FileSummary] = []
        total_bytes = 0
        skipped_binary = 0
        skipped_limits = 0
        scanned = 0

        for path in root.rglob("*"):
            if not path.is_file():
                continue
            scanned += 1
            rel_path = path.relative_to(root).as_posix()
            if not _apply_glob_filters(rel_path, include_globs, exclude_globs):
                continue
            size = path.stat().st_size
            if len(files) >= max_files or total_bytes + size > max_total_bytes:
                skipped_limits += 1
                continue
            text = _safe_read_text(path)
            if not text:
                skipped_binary += 1
                continue
            ext = path.suffix.lower()
            symbols = _extract_symbols(text, ext)
            snippet = "\n".join(text.splitlines()[:16])[:1200]
            lower_text = text.lower()
            role_tags = _infer_role_tags(rel_path, lower_text, ext)
            method_score = _score_by_keywords(lower_text, METHOD_KEYWORDS)
            experiment_score = _score_by_keywords(lower_text, EXPERIMENT_KEYWORDS)
            result_score = _score_by_keywords(lower_text, RESULT_KEYWORDS)
            evidence_strength = (
                method_score
                + experiment_score
                + result_score
                + len(role_tags)
                + _readme_priority_boost(rel_path)
            )
            summary = _build_summary(rel_path, symbols, text, role_tags)
            files.append(
                FileSummary(
                    path=rel_path,
                    extension=ext,
                    size=size,
                    symbols=symbols,
                    summary=summary,
                    snippet=snippet,
                    lower_text=lower_text,
                    method_score=method_score,
                    experiment_score=experiment_score,
                    result_score=result_score,
                    role_tags=role_tags,
                    evidence_strength=evidence_strength,
                )
            )
            total_bytes += size

        files.sort(key=lambda x: (x.evidence_strength, x.size), reverse=True)
        return files, {
            "scanned_files": scanned,
            "indexed_files": len(files),
            "indexed_total_bytes": total_bytes,
            "skipped_binary_or_unreadable": skipped_binary,
            "skipped_by_limits": skipped_limits,
        }

    def _to_evidence_pack(self, files: List[FileSummary], section_type: str) -> List[Dict[str, Any]]:
        output: List[Dict[str, Any]] = []
        for idx, file_summary in enumerate(files, start=1):
            output.append(
                {
                    "evidence_id": f"{section_type[:3].upper()}_{idx:03d}",
                    "path": file_summary.path,
                    "symbol": ", ".join(file_summary.symbols[:3]),
                    "snippet": file_summary.snippet,
                    "why_relevant": f"Likely supports {section_type} writing objectives",
                    "confidence": _evidence_confidence(file_summary.evidence_strength),
                }
            )
        return output

    def _build_code_evidence_graph(self, files: List[FileSummary], max_nodes: int = 40) -> List[Dict[str, Any]]:
        graph: List[Dict[str, Any]] = []
        for idx, file_summary in enumerate(files[:max_nodes], start=1):
            dominant = max(
                [("method", file_summary.method_score), ("experiment", file_summary.experiment_score), ("result", file_summary.result_score)],
                key=lambda x: x[1],
            )[0]
            graph.append(
                {
                    "evidence_id": f"EV{idx:03d}",
                    "path": file_summary.path,
                    "symbols": file_summary.symbols[:6],
                    "role_tags": file_summary.role_tags,
                    "dominant_role": dominant,
                    "purpose": file_summary.summary,
                    "snippet": file_summary.snippet[:800],
                    "confidence": _evidence_confidence(file_summary.evidence_strength),
                }
            )
        return graph

    def _build_writing_assets(
        self,
        method_files: List[FileSummary],
        experiment_files: List[FileSummary],
        result_files: List[FileSummary],
    ) -> Dict[str, List[Dict[str, Any]]]:
        def _asset_rows(files: List[FileSummary], prefix: str, max_items: int) -> List[Dict[str, Any]]:
            rows: List[Dict[str, Any]] = []
            for idx, file_summary in enumerate(files[:max_items], start=1):
                rows.append(
                    {
                        "asset_id": f"{prefix}_{idx:02d}",
                        "title": file_summary.summary,
                        "details": f"Use {file_summary.path} as implementation evidence.",
                        "paths": [file_summary.path],
                        "symbols": file_summary.symbols[:4],
                    }
                )
            return rows

        limitations: List[Dict[str, Any]] = []
        for file_summary in (method_files + experiment_files + result_files)[:8]:
            if "risk_signal" in file_summary.role_tags:
                limitations.append(
                    {
                        "asset_id": f"RISK_{len(limitations)+1:02d}",
                        "title": f"Potential limitation in {file_summary.path}",
                        "details": "Contains TODO/warning/limitation signals that should be discussed carefully.",
                        "paths": [file_summary.path],
                    }
                )
        return {
            "method_pipeline": _asset_rows(method_files, "METHOD", 8),
            "experiment_protocol": _asset_rows(experiment_files, "EXP", 8),
            "result_readouts": _asset_rows(result_files, "RESULT", 8),
            "risk_limitations": limitations[:5],
        }

    def _build_section_asset_packs(
        self,
        evidence_graph: List[Dict[str, Any]],
        writing_assets: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Dict[str, Any]]:
        evidence_ids_by_role: Dict[str, List[str]] = {"method": [], "experiment": [], "result": []}
        overview_ids: List[str] = []
        for ev in evidence_graph:
            role = ev.get("dominant_role", "method")
            if role not in evidence_ids_by_role:
                continue
            evidence_ids_by_role[role].append(ev.get("evidence_id", ""))
            role_tags = ev.get("role_tags", []) or []
            if "project_overview" in role_tags and ev.get("evidence_id"):
                overview_ids.append(ev.get("evidence_id", ""))

        packs: Dict[str, Dict[str, Any]] = {
            "introduction": {
                "evidence_ids": overview_ids[:2] + evidence_ids_by_role["method"][:2] + evidence_ids_by_role["experiment"][:2],
                "writing_assets": [a.get("title", "") for a in writing_assets.get("method_pipeline", [])[:2]],
                "claim_guardrails": [
                    "Only describe capabilities that are traceable to code evidence IDs.",
                    "Do not claim benchmark superiority unless result evidence supports it.",
                ],
            },
            "method": {
                "evidence_ids": evidence_ids_by_role["method"][:8],
                "writing_assets": [a.get("title", "") for a in writing_assets.get("method_pipeline", [])[:6]],
                "claim_guardrails": [
                    "Method claims must be grounded in implementation details from listed evidence.",
                ],
            },
            "experiment": {
                "evidence_ids": evidence_ids_by_role["experiment"][:8],
                "writing_assets": [a.get("title", "") for a in writing_assets.get("experiment_protocol", [])[:6]],
                "claim_guardrails": [
                    "Experiment protocol claims must map to scripts/config-related evidence.",
                ],
            },
            "result": {
                "evidence_ids": evidence_ids_by_role["result"][:8],
                "writing_assets": [a.get("title", "") for a in writing_assets.get("result_readouts", [])[:6]],
                "claim_guardrails": [
                    "Result interpretation must be tied to metric/result evidence.",
                ],
            },
            "discussion": {
                "evidence_ids": evidence_ids_by_role["result"][:4] + evidence_ids_by_role["experiment"][:2],
                "writing_assets": [a.get("title", "") for a in writing_assets.get("risk_limitations", [])[:4]],
                "claim_guardrails": [
                    "Discuss limitations if risk signals exist; avoid unsupported causal language.",
                ],
            },
        }
        packs["related_work"] = packs["introduction"]
        packs["conclusion"] = packs["discussion"]
        packs["abstract"] = packs["introduction"]
        return packs

    def _build_claim_support_candidates(self, evidence_graph: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        for ev in evidence_graph[:18]:
            role = ev.get("dominant_role", "method")
            evidence_id = ev.get("evidence_id", "")
            if role == "method":
                claim = "Implementation details support the proposed method pipeline."
            elif role == "experiment":
                claim = "Experimental setup is reproducible via configuration and scripts."
            else:
                claim = "Reported outcomes are supported by in-repo metric/analysis logic."
            candidates.append(
                {
                    "claim": claim,
                    "support_evidence_ids": [evidence_id] if evidence_id else [],
                    "reason": ev.get("purpose", ""),
                }
            )
        return candidates


def format_code_context_for_prompt(
    context: Optional[Dict[str, Any]],
    section_type: str,
    retrieved_evidence: Optional[List[Dict[str, Any]]] = None,
    top_k: int = 6,
) -> str:
    if not context:
        return ""
    alias = _section_alias(section_type)
    packs = context.get("section_asset_packs", {}) or {}
    pack = packs.get(alias, {})
    graph = context.get("code_evidence_graph", []) or []
    graph_lookup = {x.get("evidence_id", ""): x for x in graph if x.get("evidence_id")}

    chosen: List[Dict[str, Any]] = []
    seen_paths = set()
    for ev_id in pack.get("evidence_ids", [])[:top_k]:
        ev = graph_lookup.get(ev_id)
        if not ev:
            continue
        path = ev.get("path", "")
        if not path or path in seen_paths:
            continue
        chosen.append(
            {
                "evidence_id": ev_id,
                "path": path,
                "symbol": ", ".join((ev.get("symbols") or [])[:3]),
                "snippet": ev.get("snippet", ""),
                "why_relevant": ev.get("purpose", ""),
            }
        )
        seen_paths.add(path)
    for ev in retrieved_evidence or []:
        if len(chosen) >= top_k:
            break
        path = ev.get("path", "")
        if not path or path in seen_paths:
            continue
        chosen.append(ev)
        seen_paths.add(path)

    writing_assets = [str(x).strip() for x in (pack.get("writing_assets", []) or []) if str(x).strip()]
    guardrails = [str(x).strip() for x in (pack.get("claim_guardrails", []) or []) if str(x).strip()]
    if not chosen and not writing_assets:
        return ""

    lines: List[str] = [
        "## Code Evidence Map",
        "(Use these as grounding for method/algorithm descriptions. "
        "Do NOT mention file names or paths in the paper text.)",
    ]
    if not chosen:
        lines.append("- No high-confidence evidence selected for this section.")
    for idx, ev in enumerate(chosen, start=1):
        ev_id = ev.get("evidence_id", f"runtime_{idx}")
        why = ev.get("why_relevant", "") or ev.get("purpose", "")
        symbol = ev.get("symbol", "")
        # Show semantic purpose only — no file paths
        desc = why if why else (f"Defines: {symbol}" if symbol else "Supporting evidence")
        lines.append(f"- `{ev_id}`: {desc}")
        if symbol and why:
            lines.append(f"  - Key symbols: {symbol}")
        snippet = (ev.get("snippet", "") or "").strip()
        if snippet:
            lines.append("  - Implementation detail:")
            lines.append("```text")
            lines.append(snippet[:700])
            lines.append("```")

    lines.append("")
    lines.append("## Section Writing Assets")
    if writing_assets:
        for item in writing_assets[:6]:
            lines.append(f"- {item}")
    else:
        lines.append("- No dedicated writing assets extracted for this section.")

    lines.append("")
    lines.append("## Claim Guardrails")
    if guardrails:
        for item in guardrails[:4]:
            lines.append(f"- {item}")
    else:
        lines.append("- Keep claims aligned with evidence IDs listed above.")
    return "\n".join(lines)


def format_code_context_for_planner(
    context: Optional[Dict[str, Any]],
    style_guide: Optional[str] = None,
    max_items_per_section: int = 3,
) -> str:
    if not context:
        return ""
    writing_assets = context.get("writing_assets", {}) or {}
    section_packs = context.get("section_asset_packs", {}) or {}
    graph = context.get("code_evidence_graph", []) or []
    lines: List[str] = ["## Code Writing Assets Brief (for Planning)"]
    if style_guide:
        lines.append(f"- Target style/venue: {style_guide}")
    lines.append(f"- Extracted evidence nodes: {len(graph)}")

    for key, label in (
        ("method_pipeline", "Method"),
        ("experiment_protocol", "Experiment"),
        ("result_readouts", "Result"),
        ("risk_limitations", "Risk/Limitations"),
    ):
        items = writing_assets.get(key, []) or []
        if not items:
            continue
        lines.append(f"- {label} assets:")
        for item in items[:max_items_per_section]:
            title = str(item.get("title", "")).strip()
            if title:
                lines.append(f"  - {title}")

    for sec in ("introduction", "method", "experiment", "result", "discussion"):
        pack = section_packs.get(sec, {})
        evidence_ids = pack.get("evidence_ids", []) or []
        if not evidence_ids:
            continue
        lines.append(f"- Section `{sec}` suggested evidence IDs: {', '.join(evidence_ids[:6])}")

    return "\n".join(lines)


def render_code_repository_summary_markdown(context: Dict[str, Any]) -> str:
    repo_info = context.get("repository_info", {})
    stats = context.get("scan_stats", {})
    overview = context.get("repo_overview", [])
    method_pack = context.get("method_pack", [])
    experiment_pack = context.get("experiment_pack", [])
    result_pack = context.get("result_pack", [])
    writing_assets = context.get("writing_assets", {}) or {}
    evidence_graph = context.get("code_evidence_graph", []) or []
    claims = context.get("claim_support_candidates", []) or []

    lines = [
        "# Code Repository Summary",
        "",
        "## Repository Info",
        f"- Type: `{repo_info.get('type', 'unknown')}`",
        f"- Source: `{repo_info.get('source', 'unknown')}`",
    ]
    if repo_info.get("ref"):
        lines.append(f"- Ref: `{repo_info.get('ref')}`")

    lines.extend(
        [
            "",
            "## Scan Stats",
            f"- Scanned files: {stats.get('scanned_files', 0)}",
            f"- Indexed files: {stats.get('indexed_files', 0)}",
            f"- Indexed bytes: {stats.get('indexed_total_bytes', 0)}",
            f"- Skipped binary/unreadable: {stats.get('skipped_binary_or_unreadable', 0)}",
            f"- Skipped by limits: {stats.get('skipped_by_limits', 0)}",
            "",
            "## Repository Overview",
        ]
    )
    for item in overview[:12]:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Writing-Oriented Assets",
            f"- Evidence graph nodes: {len(evidence_graph)}",
            f"- Claim support candidates: {len(claims)}",
        ]
    )
    for key, label in (
        ("method_pipeline", "Method"),
        ("experiment_protocol", "Experiment"),
        ("result_readouts", "Result"),
        ("risk_limitations", "Risk/Limitations"),
    ):
        rows = writing_assets.get(key, []) or []
        lines.append(f"- {label} assets: {len(rows)}")
        for row in rows[:4]:
            lines.append(f"  - {row.get('title', '')}")

    def _add_pack(title: str, pack: List[Dict[str, Any]]) -> None:
        lines.append("")
        lines.append(f"## {title}")
        if not pack:
            lines.append("- No strong evidence extracted.")
            return
        for ev in pack[:8]:
            path = ev.get("path", "")
            symbol = ev.get("symbol", "")
            why = ev.get("why_relevant", "")
            lines.append(f"- `{path}`")
            if symbol:
                lines.append(f"  - Symbols: {symbol}")
            if why:
                lines.append(f"  - Why relevant: {why}")

    _add_pack("Method Evidence Index", method_pack)
    _add_pack("Experiment Evidence Index", experiment_pack)
    _add_pack("Result Evidence Index", result_pack)

    lines.extend(
        [
            "",
            "## Known Limitations",
            "- Ranking remains heuristic and should be treated as guidance, not proof.",
            "- Extremely large repositories may be partially indexed due to configured limits.",
            "- Binary assets are ignored during text understanding.",
        ]
    )
    return "\n".join(lines).strip() + "\n"
