# Google Colab workflow

The notebooks are optional. For plain Colab cells, use
`!python scripts/check_environment.py`, `!python scripts/build_pipeline.py`,
and `!python scripts/run_retrieval.py ...` as documented in the repository
README. If using notebooks, run them in order:

1. `00_colab_setup.ipynb` mounts Drive, installs the editable package, and puts
   all data, model caches, artifacts, and outputs on Drive.
2. `01_ingest_index.ipynb` uses supplied AIC keyframes/metadata first, then
   builds resumable visual and optional OCR/ASR indexes in batches.
3. `02_search_evaluate.ipynb` runs KIS, TRAKE, and grounded QA examples,
   exports canonical JSONL/CSV, and validates them.

Set `DATA_PATH` to the folder containing the mounted AIC2025 benchmark
(`AIC2025_ROOT` remains a legacy fallback). The expected keyframe location is an
editable variable rather than a hard-coded Drive path. Re-running an unchanged
index command returns `reused: true` and reloads vectors/corpora from Drive
without encoding images again.

The setup notebook loads the repository `.env` automatically. Because Git
ignores that file, create it from `.env.example` after cloning, or set
`HCM_AI_ENV_FILE` before the first `hcm_ai` import when the private file lives
on Drive. Existing Colab/runtime variables take precedence over dotenv values.

Gemini is optional. Prefer adding `GOOGLE_API_KEY` in Colab Secrets instead of
a notebook or dotenv file, then change the planner/answerer switches in the
search notebook. With no key, the heuristic planner stays local and QA returns
`answer=null` with retrieved evidence instead of guessing.

The notebook shell commands call the thin scripts in `scripts/`; reusable
pipeline logic remains under `src/hcm_ai/`.
