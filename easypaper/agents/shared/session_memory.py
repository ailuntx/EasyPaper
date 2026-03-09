"""
Session Memory - Shared state for a single paper generation session.
- **Description**:
    - Provides a unified memory object for cross-agent coordination
    - Stores plan, generated sections, review history, and agent logs
    - Review parts persist to files in the output directory
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import hashlib

from pydantic import BaseModel, Field

logger = logging.getLogger("uvicorn.error")


# =========================================================================
# Data models
# =========================================================================

class ReviewEntry(BaseModel):
    """Human-readable review log entry (one per section per iteration)."""
    iteration: int
    target: str
    review_comment: str
    feedback: str


class ReviewRecord(BaseModel):
    """Internal record of a single review iteration (kept for orchestration)."""
    iteration: int
    reviewer: str
    timestamp: str = ""
    passed: bool = False
    feedback_summary: str = ""
    section_feedbacks: Dict[str, Any] = Field(default_factory=dict)
    hierarchical_feedbacks: List[Dict[str, Any]] = Field(default_factory=list)
    agent_feedbacks: Dict[str, Dict[str, List[Dict[str, Any]]]] = Field(default_factory=dict)
    decision_trace: List[Dict[str, Any]] = Field(default_factory=list)
    revision_plan: List[Dict[str, Any]] = Field(default_factory=list)
    before_after_semantic_check: List[Dict[str, Any]] = Field(default_factory=list)
    conflict_resolution: List[Dict[str, Any]] = Field(default_factory=list)
    baseline_gap_audit: Dict[str, Any] = Field(default_factory=dict)
    issue_lifecycle: List[Dict[str, Any]] = Field(default_factory=list)
    writer_response_section: List[Dict[str, Any]] = Field(default_factory=list)
    writer_response_paragraph: List[Dict[str, Any]] = Field(default_factory=list)
    reviewer_verification: List[Dict[str, Any]] = Field(default_factory=list)
    regression_report: Dict[str, Any] = Field(default_factory=dict)
    actions_taken: List[str] = Field(default_factory=list)
    result_snapshot: Dict[str, int] = Field(default_factory=dict)

    def to_review_entries(self) -> List[ReviewEntry]:
        """Flatten this record into human-readable ReviewEntry items."""
        entries: List[ReviewEntry] = []
        for section, fb in self.section_feedbacks.items():
            if not isinstance(fb, dict):
                continue
            action = fb.get("action", "ok")
            comment_parts: List[str] = []
            if self.feedback_summary:
                comment_parts.append(self.feedback_summary)
            para_fbs = fb.get("paragraph_feedbacks", [])
            for pf in para_fbs:
                if isinstance(pf, dict) and pf.get("feedback"):
                    comment_parts.append(pf["feedback"])
            review_comment = "; ".join(comment_parts) if comment_parts else "No issues."
            revised = section in {
                a.replace("revised:", "") for a in self.actions_taken
            }
            if revised:
                feedback_str = f"Revised ({action})"
            elif action == "ok":
                feedback_str = "Passed, no changes needed"
            else:
                feedback_str = f"Flagged ({action}), not revised this iteration"
            entries.append(ReviewEntry(
                iteration=self.iteration,
                target=section,
                review_comment=review_comment[:500],
                feedback=feedback_str,
            ))
        if not entries and self.feedback_summary:
            entries.append(ReviewEntry(
                iteration=self.iteration,
                target="overall",
                review_comment=self.feedback_summary[:500],
                feedback="passed" if self.passed else "issues found",
            ))
        return entries

    def to_iteration_export(self) -> Dict[str, Any]:
        """
        Build iteration-centric hierarchical export payload.
        - **Description**:
            - Groups feedback by level (document/section/paragraph/sentence)
            - Each feedback item explicitly includes source_agent/source_stage
            - Preserves actions and a compact result snapshot
        """
        level_buckets: Dict[str, List[Dict[str, Any]]] = {
            "document_feedbacks": [],
            "section_feedbacks": [],
            "paragraph_feedbacks": [],
            "sentence_feedbacks": [],
        }

        # Primary source: grouped agent feedbacks assembled by metadata agent.
        if self.agent_feedbacks:
            for agent_name, grouped in self.agent_feedbacks.items():
                if not isinstance(grouped, dict):
                    continue
                for bucket in level_buckets.keys():
                    for raw_item in (grouped.get(bucket, []) or []):
                        if not isinstance(raw_item, dict):
                            continue
                        item = dict(raw_item)
                        source_agent = str(
                            item.get("source_agent")
                            or item.get("agent")
                            or item.get("checker")
                            or agent_name
                            or "reviewer"
                        )
                        source_stage = str(item.get("source_stage") or source_agent)
                        item["source_agent"] = source_agent
                        item["source_stage"] = source_stage
                        item.setdefault("agent", source_agent)
                        level_buckets[bucket].append(item)

        # Fallback source: flat hierarchical feedbacks.
        if not any(level_buckets.values()) and self.hierarchical_feedbacks:
            for raw_item in self.hierarchical_feedbacks:
                if not isinstance(raw_item, dict):
                    continue
                item = dict(raw_item)
                level = str(item.get("level", "section")).strip().lower()
                bucket = f"{level}_feedbacks"
                if bucket not in level_buckets:
                    bucket = "section_feedbacks"
                source_agent = str(
                    item.get("source_agent")
                    or item.get("agent")
                    or item.get("checker")
                    or "reviewer"
                )
                source_stage = str(item.get("source_stage") or source_agent)
                item["source_agent"] = source_agent
                item["source_stage"] = source_stage
                item.setdefault("agent", source_agent)
                level_buckets[bucket].append(item)

        return {
            "iteration": self.iteration,
            "reviewer": self.reviewer,
            "timestamp": self.timestamp,
            "passed": self.passed,
            "summary": self.feedback_summary,
            "document_feedbacks": level_buckets["document_feedbacks"],
            "section_feedbacks": level_buckets["section_feedbacks"],
            "paragraph_feedbacks": level_buckets["paragraph_feedbacks"],
            "sentence_feedbacks": level_buckets["sentence_feedbacks"],
            "decision_trace": self.decision_trace,
            "revision_plan": self.revision_plan,
            "before_after_semantic_check": self.before_after_semantic_check,
            "conflict_resolution": self.conflict_resolution,
            "baseline_gap_audit": self.baseline_gap_audit,
            "issue_lifecycle": self.issue_lifecycle,
            "writer_response_section": self.writer_response_section,
            "writer_response_paragraph": self.writer_response_paragraph,
            "reviewer_verification": self.reviewer_verification,
            "regression_report": self.regression_report,
            "actions_taken": self.actions_taken,
            "result_snapshot": self.result_snapshot,
        }


class AgentLogEntry(BaseModel):
    """Log entry for agent activity tracking."""
    agent: str
    phase: str
    timestamp: str = ""
    action: str = ""
    narrative: str = ""
    communication: Optional[Dict[str, Any]] = None
    details: Dict[str, Any] = Field(default_factory=dict)


# =========================================================================
# Session Memory
# =========================================================================

class SessionMemory:
    """
    Shared memory for one paper generation session.
    - **Description**:
        - Created at the start of generate_paper()
        - Passed to all phase methods for cross-agent coordination
        - Review history and logs are persisted to disk at the end
    """

    def __init__(self) -> None:
        self.plan: Optional[Any] = None
        self.generated_sections: Dict[str, str] = {}
        self.contributions: List[str] = []
        self.review_history: List[ReviewRecord] = []
        self.agent_logs: List[AgentLogEntry] = []
        self.issue_store: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def get_section(self, section_type: str) -> Optional[str]:
        """Get generated content for a section."""
        return self.generated_sections.get(section_type)

    def get_latest_review(self) -> Optional[ReviewRecord]:
        """Get the most recent review record."""
        return self.review_history[-1] if self.review_history else None

    def get_review_history_for_section(
        self, section_type: str,
    ) -> List[ReviewRecord]:
        """Get review records that mention a specific section."""
        return [
            r for r in self.review_history
            if section_type in r.section_feedbacks
        ]

    def has_been_revised(self, section_type: str) -> bool:
        """Check whether a section has been revised in any review iteration."""
        for record in self.review_history:
            if section_type in record.section_feedbacks:
                fb = record.section_feedbacks[section_type]
                if isinstance(fb, dict) and fb.get("action") not in ("ok", None):
                    return True
        return False

    def get_revision_count(self, section_type: str) -> int:
        """Count how many times a section has been revised."""
        count = 0
        for record in self.review_history:
            if section_type in record.section_feedbacks:
                fb = record.section_feedbacks[section_type]
                if isinstance(fb, dict) and fb.get("action") not in ("ok", None):
                    count += 1
        return count

    # ------------------------------------------------------------------
    # Update interface
    # ------------------------------------------------------------------

    def update_section(self, section_type: str, content: str) -> None:
        """Store or update generated content for a section."""
        self.generated_sections[section_type] = content

    def add_review(self, record: ReviewRecord) -> None:
        """Append a review record."""
        if not record.timestamp:
            record.timestamp = datetime.now().isoformat()
        self.review_history.append(record)

    @staticmethod
    def _make_issue_id(item: Dict[str, Any]) -> str:
        """
        Builds a stable issue identifier from normalized fields.
        - **Description**:
         - Uses target, issue type, checker/source, and message fingerprint to deduplicate issues across iterations.

        - **Args**:
         - `item` (Dict[str, Any]): Hierarchical feedback item.

        - **Returns**:
         - `issue_id` (str): Stable hashed issue id.
        """
        target_id = str(item.get("target_id", "document"))
        issue_type = str(item.get("issue_type", "issue")).lower()
        checker = str(item.get("checker", item.get("source_agent", "reviewer"))).lower()
        message = str(item.get("message", "")).strip().lower()
        msg_digest = hashlib.sha1(message.encode("utf-8")).hexdigest()[:10]
        return f"{target_id}|{issue_type}|{checker}|{msg_digest}"

    @staticmethod
    def _infer_locked_mode(item: Dict[str, Any]) -> str:
        """
        Infers lock mode for issue stability policy.
        - **Returns**:
         - `locked_mode` (str): `hard` for logic/fact/citation correctness, else `soft`.
        """
        issue_type = str(item.get("issue_type", "")).lower()
        message = str(item.get("message", "")).lower()
        hard_keywords = [
            "logic", "contradiction", "fact", "citation", "reference",
            "broken", "invalid", "latex", "compile",
        ]
        if any(k in issue_type for k in hard_keywords) or any(k in message for k in hard_keywords):
            return "hard"
        return "soft"

    def update_issue_lifecycle(
        self,
        iteration: int,
        hierarchical_feedbacks: List[Dict[str, Any]],
        writer_response_section: Optional[List[Dict[str, Any]]] = None,
        writer_response_paragraph: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Updates issue lifecycle state from current iteration signals.
        - **Description**:
         - Creates or updates issue records for current feedback.
         - Marks unresolved prior issues as resolved/regressed according to writer outcomes.
         - Produces iteration-level lifecycle delta and regression report.

        - **Args**:
         - `iteration` (int): Current review iteration.
         - `hierarchical_feedbacks` (List[Dict[str, Any]]): Current review feedback items.
         - `writer_response_section` (Optional[List[Dict[str, Any]]]): Section-level writer responses.
         - `writer_response_paragraph` (Optional[List[Dict[str, Any]]): Paragraph-level writer responses.

        - **Returns**:
         - `result` (Dict[str, Any]): Contains lifecycle events and regression counters.
        """
        current_ids: set = set()
        lifecycle_events: List[Dict[str, Any]] = []
        section_resp = writer_response_section or []
        para_resp = writer_response_paragraph or []

        for raw in hierarchical_feedbacks or []:
            if not isinstance(raw, dict):
                continue
            issue_id = self._make_issue_id(raw)
            current_ids.add(issue_id)
            locked_mode = self._infer_locked_mode(raw)
            existing = self.issue_store.get(issue_id)
            if existing is None:
                rec = {
                    "issue_id": issue_id,
                    "target_id": str(raw.get("target_id", "document")),
                    "section_type": str(raw.get("section_type") or ""),
                    "level": str(raw.get("level", "section")),
                    "source_agent": str(raw.get("source_agent") or raw.get("agent") or "reviewer"),
                    "checker": str(raw.get("checker") or ""),
                    "issue_type": str(raw.get("issue_type") or "issue"),
                    "message": str(raw.get("message") or ""),
                    "severity": str(raw.get("severity") or "warning"),
                    "status": "open",
                    "first_iteration": iteration,
                    "last_iteration": iteration,
                    "locked_mode": locked_mode,
                    "history": [{"iteration": iteration, "status": "open"}],
                }
                self.issue_store[issue_id] = rec
                lifecycle_events.append({
                    "issue_id": issue_id,
                    "event": "created",
                    "status": "open",
                    "locked_mode": locked_mode,
                })
            else:
                previous_status = str(existing.get("status", "open"))
                new_status = "regressed" if previous_status == "resolved" else "open"
                existing["status"] = new_status
                existing["last_iteration"] = iteration
                existing["history"].append({"iteration": iteration, "status": new_status})
                lifecycle_events.append({
                    "issue_id": issue_id,
                    "event": "seen_again",
                    "status": new_status,
                    "from_status": previous_status,
                    "locked_mode": existing.get("locked_mode", locked_mode),
                })

        handled_targets = {
            str(r.get("target_id") or r.get("section_type") or "")
            for r in section_resp + para_resp
            if isinstance(r, dict)
        }
        reopened_hard = 0
        for issue_id, rec in self.issue_store.items():
            if rec.get("last_iteration") == iteration:
                continue
            target = str(rec.get("target_id", ""))
            section_type = str(rec.get("section_type", ""))
            if target in handled_targets or section_type in handled_targets:
                prev_status = str(rec.get("status", "open"))
                rec["status"] = "resolved"
                rec["history"].append({"iteration": iteration, "status": "resolved"})
                lifecycle_events.append({
                    "issue_id": issue_id,
                    "event": "resolved_by_writer",
                    "from_status": prev_status,
                    "status": "resolved",
                    "locked_mode": rec.get("locked_mode", "soft"),
                })
                if prev_status == "regressed" and rec.get("locked_mode") == "hard":
                    reopened_hard += 1

        unresolved = sum(1 for rec in self.issue_store.values() if rec.get("status") in ("open", "regressed", "in_progress"))
        reopened_total = sum(1 for event in lifecycle_events if event.get("status") == "regressed")
        hard_violations = sum(
            1 for event in lifecycle_events
            if event.get("status") == "regressed" and event.get("locked_mode") == "hard"
        )
        regression_report = {
            "iteration": iteration,
            "created_issues": sum(1 for e in lifecycle_events if e.get("event") == "created"),
            "reopened_issues": reopened_total,
            "hard_lock_violations": hard_violations,
            "resolved_issues": sum(1 for e in lifecycle_events if e.get("event") == "resolved_by_writer"),
            "unresolved_issues": unresolved,
            "recovered_hard_regressions": reopened_hard,
        }
        return {
            "issue_lifecycle": lifecycle_events,
            "regression_report": regression_report,
        }

    def get_issue_context(self, limit: int = 40) -> Dict[str, Any]:
        """
        Returns issue-memory context for reviewer/writer prompts.
        - **Args**:
         - `limit` (int): Maximum issues to include.
        - **Returns**:
         - `context` (Dict[str, Any]): Unresolved/resolved issue summaries.
        """
        all_items = list(self.issue_store.values())
        unresolved = [x for x in all_items if x.get("status") in ("open", "regressed", "in_progress")]
        resolved = [x for x in all_items if x.get("status") == "resolved"]
        unresolved = sorted(unresolved, key=lambda x: (x.get("last_iteration", 0), x.get("severity", "")), reverse=True)[:limit]
        resolved = sorted(resolved, key=lambda x: x.get("last_iteration", 0), reverse=True)[: max(1, limit // 3)]
        return {
            "unresolved_issues": unresolved,
            "recent_resolved_issues": resolved,
            "issue_store_size": len(all_items),
        }

    def log(
        self,
        agent: str,
        phase: str,
        action: str,
        narrative: str = "",
        communication: Optional[Dict[str, Any]] = None,
        **details: Any,
    ) -> None:
        """Log an agent activity with optional human-readable narrative."""
        self.agent_logs.append(AgentLogEntry(
            agent=agent,
            phase=phase,
            timestamp=datetime.now().isoformat(),
            action=action,
            narrative=narrative,
            communication=communication,
            details=details,
        ))

    # ------------------------------------------------------------------
    # Context generation for LLM prompts
    # ------------------------------------------------------------------

    def get_writing_context(self, section_type: str) -> str:
        """
        Build structured context for writing a specific section.
        - **Description**:
            - Aggregates plan guidance, cross-section summaries, and
              contributions into a text block suitable for LLM prompts.

        - **Args**:
            - `section_type` (str): The section being written

        - **Returns**:
            - `context` (str): Multi-line context string
        """
        parts: List[str] = []

        # Plan guidance for this section
        if self.plan is not None:
            for sp in getattr(self.plan, "sections", []):
                if getattr(sp, "section_type", None) == section_type:
                    parts.append("## Plan Guidance for This Section")
                    if getattr(sp, "writing_guidance", ""):
                        parts.append(f"Writing guidance: {sp.writing_guidance}")
                    paras = getattr(sp, "paragraphs", [])
                    if paras:
                        parts.append(f"Expected paragraphs: {len(paras)}")
                        for idx, p in enumerate(paras, 1):
                            kp = getattr(p, "key_point", "")
                            role = getattr(p, "role", "")
                            sents = getattr(p, "approx_sentences", 0)
                            parts.append(
                                f"  P{idx}: [{role}] {kp} (~{sents} sentences)"
                            )
                    break

        # Contributions discovered so far
        if self.contributions:
            parts.append("## Key Contributions")
            for c in self.contributions:
                parts.append(f"- {c}")

        # Summaries of already-written sections
        summaries = self._build_section_summaries(exclude=section_type)
        if summaries:
            parts.append("## Already-Written Sections")
            parts.append(summaries)

        return "\n".join(parts) if parts else ""

    def get_revision_context(self, section_type: str) -> str:
        """
        Build context for revising a section based on review history.
        - **Description**:
            - Provides revision count, prior feedback summaries, and
              known unresolved issues to prevent regression.

        - **Args**:
            - `section_type` (str): The section being revised

        - **Returns**:
            - `context` (str): Multi-line revision history string
        """
        history = self.get_review_history_for_section(section_type)
        if not history:
            return ""

        rev_count = self.get_revision_count(section_type)
        parts: List[str] = [
            f"## Revision History for '{section_type}' (revised {rev_count} time(s) so far)"
        ]

        for rec in history:
            fb = rec.section_feedbacks.get(section_type, {})
            summary_bits: List[str] = []
            if isinstance(fb, dict):
                if fb.get("message"):
                    summary_bits.append(fb["message"])
                for pf in fb.get("paragraph_feedbacks", []):
                    if isinstance(pf, dict) and pf.get("feedback"):
                        summary_bits.append(
                            f"  - P{pf.get('paragraph_index', '?')}: {pf['feedback']}"
                        )
            if summary_bits:
                parts.append(
                    f"Iteration {rec.iteration} ({rec.reviewer}): "
                    + "; ".join(summary_bits[:6])
                )

        parts.append(
            "IMPORTANT: Do NOT regress on issues already fixed in earlier revisions."
        )
        return "\n".join(parts)

    def get_cross_section_summary(self) -> str:
        """
        Return a compact summary of all written sections.
        - **Description**:
            - Shows first and last sentence + word count for each section
            - Useful for synthesis sections (abstract, conclusion)

        - **Returns**:
            - `summary` (str): Multi-line summary string
        """
        return self._build_section_summaries()

    def to_review_context_dict(self) -> Dict[str, Any]:
        """
        Build a serializable memory snapshot for HTTP-based agents.
        - **Description**:
            - Contains plan section summaries, review history digest,
              per-section word counts, and contributions

        - **Returns**:
            - `context` (dict): JSON-serializable dictionary
        """
        # Plan section overview
        plan_sections: List[Dict[str, Any]] = []
        if self.plan is not None:
            for sp in getattr(self.plan, "sections", []):
                plan_sections.append({
                    "section_type": getattr(sp, "section_type", ""),
                    "num_paragraphs": len(getattr(sp, "paragraphs", [])),
                    "estimated_words": (
                        sp.get_estimated_words()
                        if hasattr(sp, "get_estimated_words") else 0
                    ),
                    "key_points": (
                        sp.get_key_points()
                        if hasattr(sp, "get_key_points") else []
                    ),
                })

        # Prior review issues (last two iterations)
        prior_issues: List[Dict[str, Any]] = []
        recent = self.review_history[-2:] if self.review_history else []
        for rec in recent:
            prior_issues.append({
                "iteration": rec.iteration,
                "reviewer": rec.reviewer,
                "passed": rec.passed,
                "feedback_summary": rec.feedback_summary,
                "actions_taken": rec.actions_taken,
            })

        # Per-section word counts
        word_counts: Dict[str, int] = {}
        for stype, content in self.generated_sections.items():
            word_counts[stype] = len(content.split())

        return {
            "plan_sections": plan_sections,
            "prior_issues": prior_issues,
            "word_counts": word_counts,
            "contributions": self.contributions,
            "issue_memory": self.get_issue_context(limit=30),
        }

    # ------------------------------------------------------------------
    # Unified search (used by AskTool)
    # ------------------------------------------------------------------

    _llm_refine: Optional[Any] = None

    def set_llm_refine(self, refine_fn) -> None:
        """
        Inject an LLM refinement callable for semantic search.
        - **Description**:
            - When set, search() uses a two-stage pipeline: rule-based
              candidate gathering followed by LLM-based semantic refinement.
            - When not set, search() falls back to rule-only candidates.

        - **Args**:
            - `refine_fn` (async callable): Signature
              ``async (question: str, context: str) -> str``
        """
        self._llm_refine = refine_fn

    async def search(self, question: str, scope: str = "all") -> str:
        """
        Two-stage search over session memory.
        - **Description**:
            - Stage 1 (rule filter): gather compact candidate snippets
              via keyword matching with strict token budgets.
            - Stage 2 (LLM refine): if an LLM refine callable has been
              injected via set_llm_refine(), pass the candidates + question
              to the LLM for semantic understanding and precise answers.
            - Falls back to Stage-1-only when no LLM is available.

        - **Args**:
            - `question` (str): Natural-language question
            - `scope` (str): "plan", "sections", "reviews",
              "contributions", or "all"

        - **Returns**:
            - `result` (str): Answer text
        """
        candidates = self._gather_candidates(question, scope)
        if not candidates:
            return ""

        if self._llm_refine is not None:
            try:
                return await self._llm_refine(question, candidates)
            except Exception as e:
                logger.warning("session_memory.llm_refine failed: %s", e)
                return candidates

        return candidates

    # ------------------------------------------------------------------
    # Stage 1: Rule-based candidate gathering (token-budgeted)
    # ------------------------------------------------------------------

    def _gather_candidates(self, question: str, scope: str = "all") -> str:
        """
        Gather compact candidate snippets via keyword matching.
        - **Description**:
            - Each scope has a strict output budget to keep total context
              under ~1000 tokens for a typical 7-section paper.
            - Plan: section_type + paragraph_count + guidance first sentence
            - Sections: first + last sentence + word count
            - Reviews: last 2 iterations, feedback_summary only
            - Contributions: full list (typically short)

        - **Args**:
            - `question` (str): The search question
            - `scope` (str): Which areas to search

        - **Returns**:
            - `text` (str): Compact candidate context
        """
        keywords = [w.lower() for w in question.split() if len(w) > 2]
        parts: List[str] = []

        if scope in ("plan", "all"):
            parts.extend(self._candidates_plan(keywords))

        if scope in ("sections", "all"):
            parts.extend(self._candidates_sections(keywords))

        if scope in ("reviews", "all"):
            parts.extend(self._candidates_reviews(keywords))

        if scope in ("contributions", "all"):
            if self.contributions:
                parts.append("## Contributions")
                for c in self.contributions:
                    parts.append(f"- {c}")

        return "\n".join(parts) if parts else ""

    def _candidates_plan(self, keywords: List[str]) -> List[str]:
        """Compact plan candidates: type + paragraph count + guidance snippet."""
        if self.plan is None:
            return []
        hits: List[str] = []
        for sp in getattr(self.plan, "sections", []):
            stype = getattr(sp, "section_type", "")
            guidance = getattr(sp, "writing_guidance", "") or ""
            paras = getattr(sp, "paragraphs", [])
            para_texts = " ".join(getattr(p, "key_point", "") for p in paras)
            full = f"{stype} {guidance} {para_texts}".lower()
            if not keywords or any(kw in full for kw in keywords):
                guidance_snippet = guidance.split(".")[0] if guidance else ""
                n_paras = len(paras)
                kp_list = ", ".join(
                    getattr(p, "key_point", "")[:60] for p in paras[:4]
                )
                line = f"- {stype}: {n_paras} paragraphs"
                if guidance_snippet:
                    line += f", guidance: \"{guidance_snippet}\""
                if kp_list:
                    line += f", key points: [{kp_list}]"
                figs = getattr(sp, "figure_placements", [])
                if figs:
                    fig_ids = ", ".join(getattr(f, "figure_id", "") for f in figs)
                    line += f", figures: [{fig_ids}]"
                hits.append(line)
        if hits:
            return ["## Plan"] + hits
        return []

    def _candidates_sections(self, keywords: List[str]) -> List[str]:
        """Compact section candidates: first + last sentence + word count."""
        hits: List[str] = []
        for stype, content in self.generated_sections.items():
            if not content.strip():
                continue
            content_lower = content.lower()
            if not keywords or any(kw in content_lower for kw in keywords):
                wc = len(content.split())
                sentences = [
                    s.strip() for s in content.replace("\n", " ").split(".")
                    if s.strip()
                ]
                first = (sentences[0][:120] + ".") if sentences else ""
                last = (sentences[-1][:120] + ".") if len(sentences) > 1 else ""
                preview = first
                if last and last != first:
                    preview += f" ... {last}"
                hits.append(f"- {stype} ({wc} words): {preview}")
        if hits:
            return ["## Sections"] + hits
        return []

    def _candidates_reviews(self, keywords: List[str]) -> List[str]:
        """Compact review candidates: last 2 iterations, summary only."""
        if not self.review_history:
            return []
        recent = self.review_history[-2:]
        hits: List[str] = []
        for rec in recent:
            rec_text = f"{rec.feedback_summary} {' '.join(rec.actions_taken)}".lower()
            if not keywords or any(kw in rec_text for kw in keywords):
                line = (
                    f"- Iter {rec.iteration} ({rec.reviewer}): "
                    f"passed={rec.passed}"
                )
                if rec.feedback_summary:
                    summary = rec.feedback_summary[:200]
                    line += f", \"{summary}\""
                hits.append(line)
        if hits:
            return ["## Reviews"] + hits
        return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_section_summaries(self, exclude: str = "") -> str:
        """Build short summaries (first+last sentence, word count) of written sections."""
        lines: List[str] = []
        for stype, content in self.generated_sections.items():
            if stype == exclude or not content.strip():
                continue
            wc = len(content.split())
            sentences = [s.strip() for s in content.replace("\n", " ").split(".") if s.strip()]
            first = (sentences[0] + ".") if sentences else ""
            last = (sentences[-1] + ".") if len(sentences) > 1 else ""
            preview = first
            if last and last != first:
                preview += f" ... {last}"
            if len(preview) > 300:
                preview = preview[:297] + "..."
            lines.append(f"- **{stype}** ({wc} words): {preview}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_top_issues(record: ReviewRecord, limit: int = 5) -> List[Dict[str, Any]]:
        issues: List[Dict[str, Any]] = []
        for item in record.hierarchical_feedbacks:
            if not isinstance(item, dict):
                continue
            message = str(item.get("message", "")).strip()
            if not message:
                continue
            issues.append({
                "target": str(item.get("target_id", "document")),
                "level": str(item.get("level", "section")),
                "source_agent": str(item.get("source_agent") or item.get("agent") or "reviewer"),
                "issue": message[:240],
            })
            if len(issues) >= limit:
                break
        if issues:
            return issues
        if record.feedback_summary:
            return [{
                "target": "document",
                "level": "document",
                "source_agent": record.reviewer,
                "issue": record.feedback_summary[:240],
            }]
        return []

    @staticmethod
    def _extract_core_tasks(record: ReviewRecord, limit: int = 6) -> List[Dict[str, Any]]:
        tasks: List[Dict[str, Any]] = []
        for raw in record.revision_plan:
            if not isinstance(raw, dict):
                continue
            instruction = str(raw.get("instruction", "")).strip()
            if not instruction:
                continue
            tasks.append({
                "target": str(raw.get("target_id") or raw.get("target") or raw.get("section_type") or "unknown"),
                "instruction": instruction[:260],
                "constraints": {
                    "preserve_claims": raw.get("preserve_claims", []),
                    "do_not_change": raw.get("do_not_change", []),
                },
                "reason": str(raw.get("rationale", ""))[:220],
            })
            if len(tasks) >= limit:
                break
        return tasks

    @staticmethod
    def _extract_risks(record: ReviewRecord, limit: int = 5) -> List[str]:
        risks: List[str] = []
        for event in record.issue_lifecycle:
            if not isinstance(event, dict):
                continue
            status = str(event.get("status", ""))
            if status in ("open", "regressed"):
                msg = str(event.get("message", "")).strip()
                target = str(event.get("target_id", ""))
                if msg:
                    risks.append(f"{target}: {msg[:180]}")
            if len(risks) >= limit:
                break
        if not risks:
            report = record.regression_report or {}
            reopened = int(report.get("reopened_count", 0) or 0)
            hard_lock = int(report.get("hard_lock_violations", 0) or 0)
            if reopened > 0:
                risks.append(f"reopened issues: {reopened}")
            if hard_lock > 0:
                risks.append(f"hard lock violations: {hard_lock}")
        return risks[:limit]

    def _build_readable_review_payload(self) -> Dict[str, Any]:
        iterations: List[Dict[str, Any]] = []
        for record in self.review_history:
            verification = [
                item for item in record.reviewer_verification
                if isinstance(item, dict)
            ]
            iteration_status = "passed"
            if any(not bool(v.get("passed", False)) for v in verification):
                iteration_status = "needs_followup"
            elif not record.passed:
                iteration_status = "issues_detected"
            iterations.append({
                "iteration": record.iteration,
                "status": iteration_status,
                "top_issues": self._extract_top_issues(record),
                "core_revision_tasks": self._extract_core_tasks(record),
                "revision_result": {
                    "reviewer_verification": verification,
                    "semantic_checks": record.before_after_semantic_check,
                    "summary": record.feedback_summary,
                },
                "open_risks": self._extract_risks(record),
            })
        return {"iterations": iterations}

    def persist_reviews(self, output_dir: Path) -> None:
        """
        Save review history as a flat, human-readable list of ReviewEntry items.

        - **Args**:
            - `output_dir` (Path): Paper output directory
        """
        output_dir = Path(output_dir)
        reviews_dir = output_dir / "logs" / "review"
        reviews_dir.mkdir(parents=True, exist_ok=True)
        path = reviews_dir / "review_history.json"
        iterations: list = [record.to_iteration_export() for record in self.review_history]
        payload = {"iterations": iterations}
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        # Backward compatibility: keep legacy flat path for downstream tools.
        legacy_path = output_dir / "review_history.json"
        legacy_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("session_memory.persisted_reviews path=%s iterations=%d", path, len(iterations))

    def persist_readable_reviews(self, output_dir: Path) -> None:
        """
        Save concise human-readable review history.

        - **Args**:
            - `output_dir` (Path): Paper output directory
        """
        output_dir = Path(output_dir)
        reviews_dir = output_dir / "logs" / "review"
        reviews_dir.mkdir(parents=True, exist_ok=True)
        path = reviews_dir / "review_history_readable.json"
        payload = self._build_readable_review_payload()
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        # Backward compatibility: keep legacy flat path for downstream tools.
        legacy_path = output_dir / "review_history_readable.json"
        legacy_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(
            "session_memory.persisted_readable_reviews path=%s iterations=%d",
            path,
            len(payload.get("iterations", [])),
        )

    def persist_logs(self, output_dir: Path) -> None:
        """
        Save agent logs to output_dir/logs/agent/agent_logs.json.

        - **Args**:
            - `output_dir` (Path): Paper output directory
        """
        output_dir = Path(output_dir)
        logs_dir = output_dir / "logs" / "agent"
        logs_dir.mkdir(parents=True, exist_ok=True)
        path = logs_dir / "agent_logs.json"
        data = [e.model_dump() for e in self.agent_logs]
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("session_memory.persisted_logs path=%s count=%d", path, len(data))

    def persist_all(self, output_dir: Path) -> None:
        """Persist both reviews and logs."""
        self.persist_reviews(output_dir)
        self.persist_readable_reviews(output_dir)
        self.persist_logs(output_dir)
