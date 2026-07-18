"""Run KIS, TRAKE, or grounded QA against persisted local FAISS/BM25 artifacts.

This is intentionally an orchestration-only command: artifacts are rebuilt by
the package's public index loaders, and all ranking behavior remains in
``SearchService``.  It always exports canonical JSONL and CSV, while a sibling
trace JSON captures the query plan and any optional-service fallback errors.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from _cli_common import configure_model_cache, default_path_from_env, emit_json, resolve_runtime_settings

from hcm_ai.artifacts import ArtifactStore
from hcm_ai.contracts import QueryRecord, TaskType
from hcm_ai.embeddings import HashEmbeddingProvider, LazyTransformersEmbeddingProvider, MarianTranslator
from hcm_ai.exporting import write_csv, write_jsonl
from hcm_ai.indexing import load_text_store, load_vector_store
from hcm_ai.ingestion import parse_txt_query
from hcm_ai.planning import GeminiQueryPlanner, GroundedAnswerer, HeuristicQueryPlanner, parse_temporal_events
from hcm_ai.reranking import Blip2ItmReranker, NullReranker, TransformersItmReranker
from hcm_ai.retrieval import SearchService
from hcm_ai.validation import validate_answer, validate_moment, validate_sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    query = parser.add_mutually_exclusive_group(required=True)
    query.add_argument("--query", help="Ad-hoc query text")
    query.add_argument("--query-file", type=Path, help="AIC TXT query file")
    parser.add_argument("--query-id", help="Stable ID for an ad-hoc query")
    parser.add_argument("--task", default="auto", choices=["auto", "KIS", "TRAKE", "QA"])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--profile", default="auto", choices=["auto", "cpu", "balanced_gpu", "paper_gpu"])
    parser.add_argument("--artifact-root", type=Path, default=Path(default_path_from_env("ARTIFACT_ROOT", "artifacts")))
    parser.add_argument(
        "--visual-index",
        action="append",
        default=[],
        metavar="NAME=FINGERPRINT",
        help="Persisted visual index from build_visual_index.py; repeat for dual retrieval",
    )
    parser.add_argument("--ocr-index", help="Persisted OCR BM25 index fingerprint")
    parser.add_argument("--asr-index", help="Persisted ASR BM25 index fingerprint")
    parser.add_argument("--visual-provider", choices=["auto", "hash", "transformers"], default="auto")
    parser.add_argument("--device", help="Torch device override for query encoders/reranker")
    parser.add_argument("--planner", choices=["heuristic", "gemini"], default="heuristic")
    parser.add_argument("--answerer", choices=["offline", "gemini"], default="offline")
    parser.add_argument("--reranker", choices=["none", "auto", "blip"], default="auto")
    parser.add_argument("--gemini-cache", type=Path, help="Drive path for bounded Gemini cache")
    parser.add_argument("--output", type=Path, required=True, help="Canonical result JSONL path")
    parser.add_argument("--csv", type=Path, help="Canonical CSV path (defaults beside --output)")
    parser.add_argument("--trace", type=Path, help="Query-plan / branch-status JSON path")
    return parser


def _parse_named_fingerprint(value: str) -> tuple[str, str]:
    name, separator, fingerprint = value.partition("=")
    if not separator or not name.strip() or not fingerprint.strip():
        raise ValueError("--visual-index must use NAME=FINGERPRINT")
    return name.strip(), fingerprint.strip()


def _provider_from_artifact(
    *,
    artifact: ArtifactStore,
    fingerprint: str,
    requested: str,
    device: str | None,
) -> Any:
    reference = artifact.get_ref("indexes/visual", fingerprint, name="vectors")
    metadata = reference.metadata.get("provider", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    class_name = str(metadata.get("class", ""))
    kind = requested
    if kind == "auto":
        kind = "hash" if class_name == "HashEmbeddingProvider" else "transformers"
    if kind == "hash":
        dimension = metadata.get("dimension")
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError(f"visual index {fingerprint} lacks a valid hash embedding dimension")
        namespace = metadata.get("namespace")
        if not isinstance(namespace, str) or not namespace:
            raise ValueError(f"visual index {fingerprint} lacks its hash embedding namespace")
        return HashEmbeddingProvider(dimension=dimension, namespace=namespace)
    if kind == "transformers":
        model_id = metadata.get("model_id")
        if not isinstance(model_id, str) or not model_id:
            raise ValueError(f"visual index {fingerprint} lacks its transformer model_id")
        return LazyTransformersEmbeddingProvider(model_id=model_id, device=device)
    raise ValueError(f"unsupported visual provider {kind!r}")


def _load_visual_branches(
    artifact: ArtifactStore,
    specifications: Sequence[str],
    *,
    requested_provider: str,
    device: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    stores: dict[str, Any] = {}
    providers: dict[str, Any] = {}
    for specification in specifications:
        name, fingerprint = _parse_named_fingerprint(specification)
        if name in stores:
            raise ValueError(f"duplicate visual branch name {name!r}")
        stores[name] = load_vector_store(artifact, fingerprint)
        providers[name] = _provider_from_artifact(
            artifact=artifact,
            fingerprint=fingerprint,
            requested=requested_provider,
            device=device,
        )
    return stores, providers


def _choose_task(text: str, fallback: TaskType) -> TaskType:
    if parse_temporal_events(text):
        return TaskType.TRAKE
    lowered = text.strip().casefold()
    if "?" in text or lowered.startswith(("ai ", "what ", "who ", "where ", "when ", "how ")):
        return TaskType.QA
    return fallback


def _load_query(args: argparse.Namespace) -> QueryRecord:
    forced = None if args.task == "auto" else TaskType(args.task)
    if args.query_file is not None:
        record = parse_txt_query(args.query_file, default_task=forced or TaskType.KIS)
        task = forced or _choose_task(record.text, record.task)
        return record.model_copy(update={"task": task})
    if not isinstance(args.query, str) or not args.query.strip():
        raise ValueError("--query must be non-empty")
    task = forced or _choose_task(args.query, TaskType.KIS)
    return QueryRecord(query_id=args.query_id or "adhoc", text=args.query, task=task)


def _planner(args: argparse.Namespace, settings: Any, cache_dir: Path) -> Any:
    translation = settings.models.catalog.get("translation", {})
    translation_model = (
        translation.get("model_id", "Helsinki-NLP/opus-mt-vi-en")
        if isinstance(translation, Mapping)
        else "Helsinki-NLP/opus-mt-vi-en"
    )
    # Marian loads only when the planner is used.  The planner catches model/
    # dependency failures and records an identity-translation fallback instead.
    baseline = HeuristicQueryPlanner(translator=MarianTranslator(model_id=translation_model))
    if args.planner == "heuristic":
        return baseline
    return GeminiQueryPlanner(
        fallback=baseline,
        api_key_env=settings.gemini.api_key_env,
        model=settings.gemini.model,
        cache_dir=cache_dir,
        max_retries=settings.gemini.max_retries,
    )


def _answerer(args: argparse.Namespace, settings: Any, cache_dir: Path) -> GroundedAnswerer:
    return GroundedAnswerer(
        api_key_env=settings.gemini.api_key_env,
        model=settings.gemini.model,
        cache_dir=cache_dir,
        max_retries=settings.gemini.max_retries if args.answerer == "gemini" else 0,
    )


def _reranker(args: argparse.Namespace, settings: Any, profile: str, device: str | None) -> Any:
    if args.reranker == "none" or profile == "cpu":
        return NullReranker()
    configured = settings.models.reranker
    if args.reranker == "auto" and not configured:
        return NullReranker()
    name = configured if args.reranker == "auto" else "blip_itm"
    entry = settings.models.catalog.get(name or "")
    if not isinstance(entry, Mapping) or not isinstance(entry.get("model_id"), str):
        return NullReranker()
    if name == "blip2_itm":
        return Blip2ItmReranker(model_id=entry["model_id"], device=device)
    return TransformersItmReranker(model_id=entry["model_id"], device=device)


def _trace_payload(service: SearchService) -> dict[str, Any]:
    trace = service.last_trace
    if trace is None:
        return {}
    return {
        "query_id": trace.query_id,
        "plan": trace.plan.model_dump(mode="json"),
        "modality_candidate_counts": trace.modality_candidate_counts,
        "branch_errors": list(trace.branch_errors),
    }


def _write_trace(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    profile, settings = resolve_runtime_settings(args.profile)
    configure_model_cache(settings.paths.model_cache)
    artifact = ArtifactStore(args.artifact_root)
    device = args.device or ("cuda" if profile != "cpu" else None)
    stores, providers = _load_visual_branches(
        artifact,
        args.visual_index,
        requested_provider=args.visual_provider,
        device=device,
    )
    ocr_store = (
        load_text_store(
            artifact,
            args.ocr_index,
            modality="ocr",
            k1=settings.indexes.ocr.k1,
            b=settings.indexes.ocr.b,
        )
        if args.ocr_index
        else None
    )
    asr_store = (
        load_text_store(
            artifact,
            args.asr_index,
            modality="asr",
            k1=settings.indexes.asr.k1,
            b=settings.indexes.asr.b,
        )
        if args.asr_index
        else None
    )
    cache_dir = args.gemini_cache or (args.artifact_root / "gemini_cache")
    service = SearchService(
        visual_stores=stores,
        visual_embeddings=providers,
        ocr_store=ocr_store,
        asr_store=asr_store,
        planner=_planner(args, settings, cache_dir),
        reranker=_reranker(args, settings, profile, device),
        answerer=_answerer(args, settings, cache_dir),
        settings=settings,
    )
    query = _load_query(args)
    if query.task == TaskType.KIS:
        records: list[Any] = service.search_moments(query, top_k=args.top_k)
        for record in records:
            validate_moment(record)
    elif query.task == TaskType.TRAKE:
        records = service.search_temporal(query, top_k=args.top_k)
        for record in records:
            validate_sequence(record)
    else:
        records = [service.answer_question(query, evidence_top_k=args.top_k)]
        validate_answer(records[0])

    jsonl_path = write_jsonl(args.output, records)
    csv_path = write_csv(args.csv or args.output.with_suffix(".csv"), records)
    trace_path = _write_trace(args.trace or args.output.with_suffix(args.output.suffix + ".trace.json"), _trace_payload(service))
    emit_json(
        {
            "task": query.task.value,
            "query_id": query.query_id,
            "result_count": len(records),
            "jsonl": jsonl_path,
            "csv": csv_path,
            "trace": trace_path,
            "branch_errors": _trace_payload(service).get("branch_errors", []),
        }
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by notebook/CLI use
    raise SystemExit(main())
