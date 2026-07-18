# HCM AI Challenge 2026 — Multimodal Moment Retrieval

Pipeline truy vấn khoảnh khắc video cho KIS, QA và TRAKE, thiết kế để chạy trên Google Colab + Google Drive. Toàn bộ logic tái sử dụng nằm trong `src/hcm_ai`; notebook và script chỉ điều phối.

Paper tham chiếu: [Unified Interactive Multimodal Moment Retrieval via Cascaded Embedding-Reranking and Temporal-Aware Score Fusion](https://arxiv.org/abs/2512.12935).

## Phạm vi đã triển khai

- FAISS-compatible cosine vector retrieval (có fallback Python) và BM25 + fuzzy matching cho OCR/ASR. Không dùng Qdrant, Elasticsearch, Docker, FastAPI hoặc web UI.
- Hợp đồng Pydantic cho video, shot, frame, OCR, ASR, query plan, moment, temporal sequence và grounded answer.
- Adapter AIC2025: đọc `.txt`, Excel (`Query Name`, `Description`, `Trans`), keyframe tree và legacy frame metadata.
- SigLIP visual index; `paper_gpu` thêm BEiT-3 index; top-K BLIP/BLIP-2 reranking được lazy-load và tự giảm cấp khi runtime không đủ khả năng.
- Query planner heuristic chạy offline; Gemini là lớp tùy chọn có JSON schema, cache/retry và fallback. Truy vấn tiếng Việt luôn được giữ để audit; nhánh visual có thể dùng Marian VI→EN.
- PaddleOCR bulk indexing, optional Gemini OCR refinement có giới hạn/caching; `faster-whisper` ASR map về frame gần nhất.
- SRRF/similarity-weighted RRF (`rrf_k=60`), min-max modality fusion, rerank candidate nhỏ, TRAKE beam search (`top-20/event`, `beam=8`, `exp(-0.01 Δt)`).
- Canonical JSONL/CSV export + validator cho KIS, TRAKE và QA. Chưa có schema nộp HCM2026 chính thức nên không đoán format submit cuối.
- Artifact fingerprint + completion manifest/checkpoint trên Drive để rerun không encode/index lại artifact hoàn tất.

## Cấu trúc

```text
configs/                 YAML profile và model/index defaults
src/hcm_ai/              package pipeline
  ingestion/             AIC2025 adapter + frame/OCR/ASR alignment
  preprocessing/         ffprobe/FFmpeg, shot, keyframe fallback
  embeddings/             SigLIP/BEiT-3/Marian + hash smoke fallback
  indexing/               FAISS-ready vector + BM25 persistence
  planning/               heuristic/Gemini plan, grounded QA
  retrieval/              SearchService public contract
  fusion/, reranking/, temporal/
scripts/                 CLI mỏng cho Colab/Drive
notebooks/               setup, ingest/index, search/evaluate/export
tests/                   unit + no-GPU smoke coverage
```

## Profiles

| Profile | Visual retrieval | Reranker | ASR | Điều kiện |
| --- | --- | --- | --- | --- |
| `cpu` | SigLIP | none | `small` | baseline CPU; hash fallback khi model dependency không có |
| `balanced_gpu` | SigLIP | BLIP ITM | `small` | GPU thường |
| `paper_gpu` | SigLIP + BEiT-3 | BLIP-2 relevance gate | `large-v3` | GPU ≥ 14 GiB và dependency/model khả dụng |

`hcm_ai.runtime.resolve_profile()` kiểm tra GPU/VRAM/dependency. `build_profile_components(..., self_test=True)` tải thử những thành phần cần thiết và giảm dần `paper_gpu → balanced_gpu → cpu` nếu model không usable.

## Cài đặt

Trên Colab, clone hoặc upload repository rồi cài package với extras phù hợp:

```bash
pip install -e ".[dev,retrieval,models,gemini,ocr,asr]"
```

Chỉ cần smoke test không GPU/API:

```bash
pip install -e ".[dev]"
```

Các optional dependency không được import khi không dùng: `faiss`, `transformers`, `PaddleOCR`, `faster-whisper`, `google-genai`, `openpyxl`.

### Cấu hình `.env`

`.env.example` được lưu trên Git, còn `.env` bị ignore nên **không xuất hiện sau khi clone**. Tạo file riêng rồi sửa các đường dẫn cho đúng Drive của bạn:

```bash
cp .env.example .env
```

Package và mọi CLI tự nạp `.env`. Thứ tự ưu tiên là: tham số CLI → biến runtime/Colab Secrets → `.env` → YAML mặc định. Vì vậy file không ghi đè biến đã được Colab cấp. Nếu giữ file riêng trên Drive, đặt `HCM_AI_ENV_FILE=/content/drive/MyDrive/.../.env` **trước lần đầu import `hcm_ai`**.

Không commit khóa. Khuyến nghị đặt `GOOGLE_API_KEY` bằng Colab Secrets; `.env` chỉ là fallback. `DATA_PATH` là thư mục dataset đầu vào, còn `DATA_ROOT` là thư mục làm việc của project. Các biến khác gồm `ARTIFACT_ROOT`, `OUTPUT_ROOT`, `MODEL_CACHE`, `AIC2025_ROOT` (alias cũ) và `HCM_AI_PROFILE`. Đặt artifact/model cache vào Drive vì `/content` là ephemeral.

`DATA_PATH` phải là đường dẫn đã mount như `/content/drive/MyDrive/AIC2025`, không phải URL Google Drive. Với folder được chia sẻ, thêm shortcut vào My Drive trước khi chạy.

## Chạy trực tiếp từng file trong cell Colab

Không bắt buộc mở các file `.ipynb`. Trong Colab, cú pháp đúng là `!python scripts/<tên_file>.py`; không phải `python--`.

Sau khi clone, chuyển vào repository, tạo `.env`, chỉnh `DATA_PATH` trong Files panel rồi cài package:

```python
%cd /content/HCM-AI-Challenge-2026
!cp -n .env.example .env
!python -m pip install -q -e ".[dev,retrieval,models,ocr,asr,gemini]"
```

Nếu dùng Gemini, đưa Colab Secret vào environment của runtime trước khi gọi file Python (không in giá trị):

```python
from google.colab import userdata
import os
gemini_key = userdata.get("GOOGLE_API_KEY")
if gemini_key:
    os.environ["GOOGLE_API_KEY"] = gemini_key
```

Kiểm tra `.env` mà không in khóa API:

```python
!python scripts/check_environment.py
```

Build baseline từ `DATA_PATH`. Lệnh tự ưu tiên frame metadata có timestamp chính xác, sau đó `keyframes/` rồi mới tới `videos/`, và ghi trạng thái vào `ARTIFACT_ROOT/pipeline_state.json`:

```python
!python scripts/build_pipeline.py
```

Nếu cấu trúc dataset không chuẩn, truyền nguồn rõ ràng:

```python
!python scripts/build_pipeline.py --keyframes-root /content/drive/MyDrive/AIC2025/keyframes
```

OCR/ASR là tùy chọn:

```python
!python scripts/build_pipeline.py --run-ocr
# Hoặc thêm: --ocr-records /duong/dan/ocr.jsonl --asr-records /duong/dan/asr.jsonl
```

Tìm kiếm sẽ tự đọc `pipeline_state.json` và tự xuất dưới `OUTPUT_ROOT`:

```python
!python scripts/run_retrieval.py \
  --query-id demo_kis --query "người đứng cạnh biển hiệu" --task KIS

!python scripts/run_retrieval.py \
  --query-id demo_trake --query $'E1: xe xuất hiện\nE2: người vẫy tay' --task TRAKE

!python scripts/run_retrieval.py \
  --query-id demo_qa --query "Biển hiệu ghi gì?" --task QA
```

Mỗi lệnh search in trường `jsonl` chứa đường dẫn kết quả. Ví dụ với `.env.example`, kiểm tra KIS bằng:

```python
!python scripts/validate_submission.py \
  --input /content/drive/MyDrive/HCM-AI-Challenge-2026/outputs/demo_kis_kis.jsonl \
  --task KIS
```

Không dùng `$DATA_PATH` hoặc `$OUTPUT_ROOT` trong lệnh shell nếu chúng chỉ nằm trong `.env`: shell mở rộng biến trước khi Python nạp dotenv. Các script trên tự đọc chúng bên trong process Python.

## Notebook tùy chọn trên Google Drive

Nếu thích giao diện notebook, có thể mở lần lượt:

1. `notebooks/00_colab_setup.ipynb` — mount Drive, set environment/cache và cài dependencies.
2. `notebooks/01_ingest_index.ipynb` — ưu tiên keyframe/metadata AIC2025 có sẵn, tạo manifest rồi index theo batch/resume.
3. `notebooks/02_search_evaluate.ipynb` — KIS, TRAKE, QA, canonical export và validator.

Các stage thấp hơn (`preprocess_videos.py`, `build_visual_index.py`,
`build_text_index.py`) vẫn có thể chạy riêng để debug. `build_pipeline.py` chỉ
điều phối các file này và lưu state; `--force` tạo build mới nhưng không xóa
artifact Drive đã hoàn tất.

## Public Python contract

```python
from hcm_ai.retrieval import SearchService

service = SearchService(...)  # inject vector/BM25 stores and optional providers

moments = service.search_moments("người đứng cạnh biển hiệu", top_k=10)
sequences = service.search_temporal("E1: xe xuất hiện\nE2: người vẫy tay", top_k=10)
answer = service.answer_question("Biển hiệu ghi gì?", evidence_top_k=5)
```

- `search_moments` trả `list[MomentResult]` với `video_id`, `shot_id` (nếu có), `frame_id`, timestamp, `image_path`, scores từng modality, fused/reranker score và provenance.
- `search_temporal` chỉ trả sequence cùng video, timestamp tăng nghiêm ngặt, không lặp frame.
- `answer_question` retrieve evidence trước. Khi Gemini unavailable/quota/schema citation lỗi, kết quả là `answer=None` cùng evidence, không đoán đáp án.
- `service.last_trace` giữ `QueryPlan`, số candidate mỗi modality và branch fallback errors để debug.

## Dữ liệu AIC2025

AIC2025 chỉ là benchmark phát triển cho HCM2026. Adapter nhận:

- query `.txt`, query workbook `.xlsx`/`.xlsm`;
- supplied keyframe tree; timestamp từ frame number/FPS hoặc metadata;
- JSON/JSONL `FrameRecord` legacy;
- raw video (FFmpeg fixed interval; TransNetV2 adapter có thể inject khi cần).

Thay dữ liệu HCM2026 bằng manifest/adapter mới, không thay public contracts hoặc retrieval logic.

## Artifact và export

`ArtifactStore` ghi JSONL + completion manifest theo fingerprint của data/model/config. Visual vectors được serialise ở định dạng portable rồi reconstruce FAISS khi runtime có package; BM25 corpus được serialise riêng. Vì vậy restart Colab không phải re-encode visual artifact đã complete.

Canonical export:

- KIS: ranked moment, modality scores, provenance.
- TRAKE: video, ordered events/timestamps, sequence/reranker score.
- QA: answer hoặc `null`, confidence, evidence moments/citations.

`scripts/validate_submission.py` kiểm tra schema, IDs, timestamp, temporal ordering, grounding và tùy chọn xác minh `image_path` tồn tại.

## Kiểm thử

```bash
python -m pytest
python -m compileall -q src tests scripts
```

Unit tests không cần GPU, Qdrant, Elasticsearch hay API key. Các test model/Colab thật là integration acceptance: index một AIC2025 subset, chạy KIS + TRAKE + QA, rồi rerun bước index và xác nhận `reused=true`.

## Lưu ý vận hành

- Không gửi toàn bộ keyframe sang Gemini. Gemini chỉ là optional query plan/grounded QA và selective OCR refinement, có cache + bounded retry/rate limit.
- Giữ Vietnamese query cho OCR/ASR; visual branch dùng bản dịch/variant bảo thủ khi available.
- Mọi đường dẫn/key/collection runtime đều thuộc YAML/environment, không hardcode máy cá nhân.
- Không commit `.env`, khóa API, dữ liệu hoặc artifact sinh ra; chỉ `.env.example` được version control.
