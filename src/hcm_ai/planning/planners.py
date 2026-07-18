"""Query planning with a deterministic offline baseline and optional Gemini.

The heuristic planner is deliberately always available.  Gemini improves query
translation and modality routing when an API key is configured, but any quota,
network, or schema failure falls back to the same serializable plan.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from hcm_ai.cache import JsonCache
from hcm_ai.contracts import QueryPlan, QueryRecord, TaskType
from hcm_ai.embeddings import IdentityTranslator, Translator


class QueryPlanner(Protocol):
    """Public interface used by :class:`hcm_ai.retrieval.SearchService`."""

    def plan(self, query: QueryRecord) -> QueryPlan: ...


_EVENT_PATTERN = re.compile(
    r"^\s*E\s*(?P<index>\d+)\s*(?::|\.|\)|-|–|—)\s*(?P<event>.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_OCR_CUES = (
    "chữ",
    "van ban",
    "văn bản",
    "text",
    "logo",
    "biển",
    "bang",
    "bảng",
    "caption",
    "phụ đề",
    "tieu de",
    "tiêu đề",
    "sign",
    "written",
)
_ASR_CUES = (
    "nói",
    "noi",
    "lời",
    "loi",
    "phát biểu",
    "phat bieu",
    "đối thoại",
    "doi thoai",
    "giọng",
    "giong",
    "bài hát",
    "bai hat",
    "speech",
    "said",
    "says",
    "hear",
    "audio",
    "song",
)


def parse_temporal_events(text: str) -> list[str]:
    """Extract ordered ``E1: …`` through ``En: …`` TRAKE events.

    Explicit indices are sorted numerically so a workbook's line wrapping does
    not alter chronology.  Repeated indices keep their first non-empty event;
    malformed text simply returns no events and lets the caller use a one-event
    fallback rather than inventing a sequence.
    """

    indexed: dict[int, str] = {}
    for match in _EVENT_PATTERN.finditer(text):
        index = int(match.group("index"))
        event = match.group("event").strip()
        if index > 0 and event and index not in indexed:
            indexed[index] = event
    return [indexed[index] for index in sorted(indexed)]


def _safe_translation(translator: Translator, text: str) -> tuple[str, str]:
    """Translate one query while preserving an offline identity fallback."""

    try:
        translated = translator.translate([text])
    except Exception:
        return text, "identity_fallback"
    if len(translated) != 1 or not isinstance(translated[0], str) or not translated[0].strip():
        return text, "identity_fallback"
    return translated[0].strip(), type(translator).__name__


def _infer_task_from_text(text: str) -> TaskType:
    if parse_temporal_events(text):
        return TaskType.TRAKE
    if "?" in text or text.strip().casefold().startswith(("ai ", "what ", "who ", "where ", "when ")):
        return TaskType.QA
    return TaskType.KIS


def _coerce_query(query: QueryRecord | str, *, task: TaskType | None = None) -> QueryRecord:
    if isinstance(query, QueryRecord):
        return query
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string or QueryRecord")
    return QueryRecord(query_id="adhoc", text=query, task=task or _infer_task_from_text(query))


def _cue_weights(query: QueryRecord, events: Sequence[str]) -> dict[str, float]:
    """Give textual cues extra weight without disabling visual evidence."""

    text = query.text.casefold()
    weights = {"visual": 0.70, "ocr": 0.15, "asr": 0.15}
    if query.task == TaskType.QA:
        weights = {"visual": 0.45, "ocr": 0.30, "asr": 0.25}
    if events:
        weights = {"visual": 0.65, "ocr": 0.15, "asr": 0.20}
    if any(cue in text for cue in _OCR_CUES):
        weights["ocr"] += 0.25
        weights["visual"] = max(0.10, weights["visual"] - 0.15)
        weights["asr"] = max(0.0, weights["asr"] - 0.10)
    if any(cue in text for cue in _ASR_CUES):
        weights["asr"] += 0.25
        weights["visual"] = max(0.10, weights["visual"] - 0.15)
        weights["ocr"] = max(0.0, weights["ocr"] - 0.10)
    return weights


def _visual_variants(translated: str) -> list[str]:
    """Return four conservative prompts without adding unobserved entities."""

    return [
        translated,
        f"video scene: {translated}",
        f"keyframe matching: {translated}",
        f"visual evidence of: {translated}",
    ]


class HeuristicQueryPlanner:
    """Fully local, deterministic planner suitable for CPU and unit tests."""

    def __init__(self, translator: Translator | None = None) -> None:
        self.translator = translator or IdentityTranslator()

    def plan(self, query: QueryRecord | str, *, task: TaskType | None = None) -> QueryPlan:
        record = _coerce_query(query, task=task)
        translated, translation_source = _safe_translation(
            self.translator,
            record.translated_text or record.text,
        )
        events = parse_temporal_events(record.text) if record.task == TaskType.TRAKE else []
        raw_weights = _cue_weights(record, events)
        return QueryPlan(
            query_id=record.query_id,
            original_query=record.text,
            translated_query=translated,
            visual_queries=_visual_variants(translated),
            ocr_query=record.text,
            asr_query=record.text,
            temporal_events=events,
            raw_weights=raw_weights,
            planner="heuristic",
            metadata={
                "task": record.task.value,
                "translation_source": translation_source,
                "original_vietnamese_query": record.text,
            },
        )


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


class GeminiQueryPlanner:
    """Best-effort Gemini planner with cache, bounded retry, and local fallback."""

    def __init__(
        self,
        *,
        fallback: HeuristicQueryPlanner | None = None,
        api_key_env: str = "GOOGLE_API_KEY",
        model: str = "gemini-2.5-flash-lite",
        cache_dir: str | Path | None = None,
        max_retries: int = 2,
        client: Any | None = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.fallback = fallback or HeuristicQueryPlanner()
        self.api_key_env = api_key_env
        self.model = model
        self.max_retries = max_retries
        self.client = client
        self.cache = JsonCache(cache_dir, namespace="gemini_query_plans") if cache_dir else None

    def _client(self) -> Any:
        if self.client is not None:
            return self.client
        api_key = os.environ.get(self.api_key_env) or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(f"missing Gemini key in {self.api_key_env}")
        try:
            from google import genai
        except ImportError as error:
            raise RuntimeError("Install hcm-ai[gemini] to enable Gemini query planning") from error
        self.client = genai.Client(api_key=api_key)
        return self.client

    def _prompt(self, record: QueryRecord, baseline: QueryPlan) -> str:
        return f"""You are the query planner for a multimodal video-moment retrieval system.
Return JSON only. Do not answer the query. Preserve the original Vietnamese text.
Use this schema exactly:
{{
  \"translated_query\": \"English visual-search translation\",
  \"visual_queries\": [\"exactly four conservative visual variants\"],
  \"ocr_query\": \"query for visible text\",
  \"asr_query\": \"query for speech\",
  \"raw_weights\": {{\"visual\": number, \"ocr\": number, \"asr\": number}},
  \"temporal_events\": [\"ordered events only for TRAKE\"]
}}
Task: {record.task.value}
Original query: {record.text}
Offline baseline (use it when uncertain): {baseline.model_dump_json()}
"""

    def _request(self, prompt: str) -> str:
        client = self._client()
        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )
        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Gemini returned no text response")
        return text

    @staticmethod
    def _merge_response(raw: Mapping[str, Any], record: QueryRecord, baseline: QueryPlan) -> QueryPlan:
        visual = raw.get("visual_queries", raw.get("visual_variants", baseline.visual_queries))
        if not isinstance(visual, list):
            visual = baseline.visual_queries
        cleaned_visual = [item.strip() for item in visual if isinstance(item, str) and item.strip()]
        if len(cleaned_visual) < 4:
            cleaned_visual.extend(item for item in baseline.visual_queries if item not in cleaned_visual)
        events = raw.get("temporal_events", baseline.temporal_events)
        if not isinstance(events, list):
            events = baseline.temporal_events
        raw_weights = raw.get("raw_weights", raw.get("weights", baseline.raw_weights))
        if not isinstance(raw_weights, Mapping):
            raw_weights = baseline.raw_weights
        return QueryPlan(
            query_id=record.query_id,
            original_query=record.text,
            translated_query=str(raw.get("translated_query", raw.get("translation", baseline.translated_query)) or baseline.translated_query),
            visual_queries=cleaned_visual[:4],
            ocr_query=str(raw.get("ocr_query", baseline.ocr_query) or baseline.ocr_query or record.text),
            asr_query=str(raw.get("asr_query", baseline.asr_query) or baseline.asr_query or record.text),
            temporal_events=[item for item in events if isinstance(item, str)],
            raw_weights=dict(raw_weights),
            planner="gemini",
            metadata={
                **baseline.metadata,
                "gemini_model": baseline.metadata.get("gemini_model"),
                "original_vietnamese_query": record.text,
            },
        )

    def _fallback(self, baseline: QueryPlan, reason: str) -> QueryPlan:
        return baseline.model_copy(
            update={
                "metadata": {**baseline.metadata, "gemini_fallback_reason": reason},
            }
        )

    def plan(self, query: QueryRecord | str, *, task: TaskType | None = None) -> QueryPlan:
        record = _coerce_query(query, task=task)
        baseline = self.fallback.plan(record)
        cache_key = {
            "kind": "query_plan",
            "model": self.model,
            "query_id": record.query_id,
            "task": record.task.value,
            "text": record.text,
        }
        if self.cache is not None:
            cached = self.cache.get(cache_key)
            if isinstance(cached, Mapping):
                try:
                    return self._merge_response(cached, record, baseline).model_copy(
                        update={"metadata": {**baseline.metadata, "cache_hit": True, "gemini_model": self.model}}
                    )
                except Exception:
                    # A malformed historical cache must never stop a search.
                    pass

        failure = "unknown Gemini failure"
        for attempt in range(self.max_retries + 1):
            try:
                raw_response = json.loads(_strip_json_fence(self._request(self._prompt(record, baseline))))
                if not isinstance(raw_response, Mapping):
                    raise ValueError("Gemini response must be a JSON object")
                plan = self._merge_response(raw_response, record, baseline).model_copy(
                    update={"metadata": {**baseline.metadata, "gemini_model": self.model}}
                )
                if self.cache is not None:
                    self.cache.set(cache_key, dict(raw_response))
                return plan
            except Exception as error:
                failure = f"{type(error).__name__}: {error}"
                if attempt < self.max_retries:
                    time.sleep(0.25 * (attempt + 1))
        return self._fallback(baseline, failure)


__all__ = [
    "GeminiQueryPlanner",
    "HeuristicQueryPlanner",
    "QueryPlanner",
    "parse_temporal_events",
]
