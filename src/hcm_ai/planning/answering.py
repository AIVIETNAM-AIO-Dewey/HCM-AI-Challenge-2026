"""Grounded QA adapter that refuses to invent answers without evidence."""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from hcm_ai.cache import JsonCache
from hcm_ai.contracts import AnswerResult, MomentResult


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


class GroundedAnswerer:
    """Use Gemini only after retrieval, returning ``answer=None`` on failure.

    The model receives a compact evidence manifest.  Its returned citations are
    checked against the retrieved frames before an answer is released, which
    keeps unavailable quota and unsupported claims from becoming fabricated QA
    output.
    """

    def __init__(
        self,
        *,
        api_key_env: str = "GOOGLE_API_KEY",
        model: str = "gemini-2.5-flash-lite",
        cache_dir: str | Path | None = None,
        max_retries: int = 2,
        client: Any | None = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.api_key_env = api_key_env
        self.model = model
        self.max_retries = max_retries
        self.client = client
        self.cache = JsonCache(cache_dir, namespace="gemini_grounded_answers") if cache_dir else None

    def _client(self) -> Any:
        if self.client is not None:
            return self.client
        api_key = os.environ.get(self.api_key_env) or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(f"missing Gemini key in {self.api_key_env}")
        try:
            from google import genai
        except ImportError as error:
            raise RuntimeError("Install hcm-ai[gemini] to enable grounded QA") from error
        self.client = genai.Client(api_key=api_key)
        return self.client

    @staticmethod
    def _manifest(evidence: Sequence[MomentResult]) -> list[dict[str, Any]]:
        return [
            {
                "video_id": item.video_id,
                "frame_id": item.frame_id,
                "timestamp": item.timestamp,
                "image_path": item.image_path,
                "provenance": item.provenance,
                "metadata": item.metadata,
            }
            for item in evidence
        ]

    def _prompt(self, question: str, evidence: Sequence[MomentResult]) -> str:
        return f"""Answer the video question only from the supplied evidence.
Return JSON only using this schema:
{{"answer": string|null, "confidence": number between 0 and 1,
  "citations": [{{"frame_id": string, "timestamp": number}}]}}
If the evidence does not establish an answer, return answer null and citations [].
Every non-null answer must cite at least one supplied frame.
Question: {question}
Evidence manifest: {json.dumps(self._manifest(evidence), ensure_ascii=False)}
"""

    def _request(self, prompt: str) -> str:
        response = self._client().models.generate_content(
            model=self.model,
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )
        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Gemini returned no text response")
        return text

    @staticmethod
    def _validated_answer(
        query_id: str,
        raw: Mapping[str, Any],
        evidence: Sequence[MomentResult],
        model: str,
    ) -> AnswerResult:
        answer = raw.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            return AnswerResult(
                query_id=query_id,
                answer=None,
                evidence=list(evidence),
                provider=f"gemini:{model}",
                error="model_declined_or_ungrounded",
            )
        citations = raw.get("citations")
        if not isinstance(citations, list):
            citations = []
        available = {(item.frame_id, float(item.timestamp)) for item in evidence}
        valid = [
            citation
            for citation in citations
            if isinstance(citation, Mapping)
            and isinstance(citation.get("frame_id"), str)
            and isinstance(citation.get("timestamp"), (int, float))
            and (citation["frame_id"], float(citation["timestamp"])) in available
        ]
        if not valid:
            return AnswerResult(
                query_id=query_id,
                answer=None,
                evidence=list(evidence),
                provider=f"gemini:{model}",
                error="model_answer_missing_valid_citation",
            )
        raw_confidence = raw.get("confidence", 0.0)
        try:
            confidence = min(1.0, max(0.0, float(raw_confidence)))
        except (TypeError, ValueError):
            confidence = 0.0
        citation_suffix = " ".join(
            f"[{citation['frame_id']} @ {float(citation['timestamp']):.3f}s]" for citation in valid
        )
        answer_with_citations = f"{answer.strip()} {citation_suffix}".strip()
        return AnswerResult(
            query_id=query_id,
            answer=answer_with_citations,
            confidence=confidence,
            evidence=list(evidence),
            provider=f"gemini:{model}",
            metadata={"citations": [dict(item) for item in valid]},
        )

    def answer(
        self,
        *,
        query_id: str,
        question: str,
        evidence: Sequence[MomentResult],
    ) -> AnswerResult:
        evidence_list = list(evidence)
        if not evidence_list:
            return AnswerResult(
                query_id=query_id,
                answer=None,
                evidence=[],
                provider="grounded_qa",
                error="no_retrieved_evidence",
            )
        key = {
            "kind": "grounded_answer",
            "model": self.model,
            "query_id": query_id,
            "question": question,
            "evidence": [(item.frame_id, item.timestamp) for item in evidence_list],
        }
        if self.cache is not None:
            cached = self.cache.get(key)
            if isinstance(cached, Mapping):
                result = self._validated_answer(query_id, cached, evidence_list, self.model)
                return result.model_copy(update={"metadata": {**result.metadata, "cache_hit": True}})

        failure = "unknown Gemini failure"
        for attempt in range(self.max_retries + 1):
            try:
                raw = json.loads(_strip_json_fence(self._request(self._prompt(question, evidence_list))))
                if not isinstance(raw, Mapping):
                    raise ValueError("Gemini response must be a JSON object")
                if self.cache is not None:
                    self.cache.set(key, dict(raw))
                return self._validated_answer(query_id, raw, evidence_list, self.model)
            except Exception as error:
                failure = f"{type(error).__name__}: {error}"
                if attempt < self.max_retries:
                    time.sleep(0.25 * (attempt + 1))
        return AnswerResult(
            query_id=query_id,
            answer=None,
            evidence=evidence_list,
            provider="grounded_qa",
            error=failure,
        )


__all__ = ["GroundedAnswerer"]
