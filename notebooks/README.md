# Google Colab workflow

Run the notebooks in order:

1. `00_colab_setup.ipynb` mounts Drive, installs the editable package, and puts
   all data, model caches, artifacts, and outputs on Drive.
2. `01_ingest_index.ipynb` uses supplied AIC keyframes/metadata first, then
   builds resumable visual and optional OCR/ASR indexes in batches.
3. `02_search_evaluate.ipynb` runs KIS, TRAKE, and grounded QA examples,
   exports canonical JSONL/CSV, and validates them.

Set `AIC2025_ROOT` in the ingestion notebook to the folder containing the
mounted AIC2025 benchmark. The expected keyframe location is shown as an
editable variable rather than a hard-coded Drive path. Re-running an unchanged
index command returns `reused: true` and reloads vectors/corpora from Drive
without encoding images again.

Gemini is optional. Add `GOOGLE_API_KEY` in Colab Secrets (not in a notebook,
`.env`, or Drive state file), then change the planner/answerer switches in the
search notebook. With no key, the heuristic planner stays local and QA returns
`answer=null` with retrieved evidence instead of guessing.

The notebook shell commands call the thin scripts in `scripts/`; reusable
pipeline logic remains under `src/hcm_ai/`.
