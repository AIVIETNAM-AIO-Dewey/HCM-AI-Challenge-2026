# HCM AI Challenge 2026 - Multimodal Moment Retrieval Pipeline

Repository này được dùng để xây lại pipeline truy vấn video cho HCM AI Challenge, lấy cảm hứng chính từ paper:

> Unified Interactive Multimodal Moment Retrieval via Cascaded Embedding-Reranking and Temporal-Aware Score Fusion  
> arXiv: https://arxiv.org/pdf/2512.12935

Mục tiêu của project là xây một hệ thống tìm kiếm khoảnh khắc trong video theo truy vấn tự nhiên, hỗ trợ các bài toán kiểu KIS, VQA và TRAKE. Pipeline tập trung vào ba ý tưởng chính:

- Truy xuất nhiều tầng: dùng dual encoder để lấy candidate nhanh, sau đó rerank bằng cross-modal model.
- Tìm kiếm đa modality: kết hợp visual, OCR và ASR thay vì chỉ dựa vào frame embedding.
- Suy luận theo thời gian: gom các frame/segment thành chuỗi sự kiện hợp lý, phạt các khoảng thời gian rời rạc bằng exponential decay.

## 1. Bài toán

Input chính:

- Tập video của cuộc thi.
- Truy vấn tiếng Việt hoặc tiếng Anh từ người dùng.
- Các loại truy vấn:
  - `KIS`: tìm frame/khoảnh khắc theo mô tả hình ảnh.
  - `VQA`: tìm bằng chứng trong video để trả lời câu hỏi.
  - `TRAKE`: tìm chuỗi sự kiện hoặc đoạn thời gian có quan hệ temporal.

Output mong muốn:

- Danh sách kết quả đã rank theo độ liên quan.
- Mỗi kết quả gồm `video_id`, `frame_id`, `timestamp`, thumbnail/keyframe, điểm từng modality và điểm fusion cuối.
- Với query temporal, output nên có sequence gồm nhiều mốc thời gian thay vì một frame đơn lẻ.

## 2. Kiến trúc tổng quan

```text
                 OFFLINE INDEXING

Videos
  |
  |-- audio extraction -----------------------> Whisper ASR
  |                                             |
  |                                             v
  |                                      ASR text segments
  |
  |-- shot detection / keyframe extraction ---> Keyframes
                                                |
                                                |-- BEiT-3 embeddings
                                                |-- SigLIP embeddings
                                                |-- Gemini/OCR text extraction
                                                |
                                                v
                                      Multimodal indexes
                                      - Qdrant: visual vectors
                                      - Elasticsearch: OCR/ASR text
                                      - Metadata store: video/frame/time


                 ONLINE RETRIEVAL

User query
  |
  |-- LLM query decomposition / expansion
  |     - visual query
  |     - OCR query
  |     - ASR query
  |     - modality weights
  |
  |-- parallel search
  |     - visual: Qdrant, BEiT-3 + SigLIP
  |     - OCR: Elasticsearch
  |     - ASR: Elasticsearch
  |
  |-- candidate fusion
  |-- BLIP-2 reranking
  |-- temporal beam search / decay scoring
  |
  v
Ranked moments / sequences
```

## 3. Pipeline offline

### 3.1 Video preprocessing

Nhiệm vụ:

- Chuẩn hóa metadata cho từng video.
- Tách audio bằng `ffmpeg`.
- Chạy shot boundary detection, ưu tiên `TransNetV2`.
- Trích keyframe đại diện cho mỗi shot, mặc định 3 keyframe/shot.

Metadata tối thiểu cho mỗi keyframe:

```json
{
  "video_id": "L01_V001",
  "shot_id": "L01_V001_S00042",
  "frame_id": "L01_V001_F001234",
  "timestamp": 52.36,
  "image_path": "data/processed/keyframes/L01_V001/F001234.jpg"
}
```

### 3.2 Visual embedding

Mỗi keyframe được encode bằng hai model:

- `BEiT-3`: ưu tiên semantic precision.
- `SigLIP`: ưu tiên retrieval coverage và khả năng generalize.

Embedding cần được normalize trước khi index. Trong Qdrant, nên dùng named vectors để lưu nhiều vector cho cùng một point:

```json
{
  "id": "L01_V001_F001234",
  "vectors": {
    "beit3": [0.01, 0.02],
    "siglip": [0.03, 0.04]
  },
  "payload": {
    "video_id": "L01_V001",
    "timestamp": 52.36,
    "image_path": "data/processed/keyframes/L01_V001/F001234.jpg"
  }
}
```

### 3.3 OCR indexing

OCR dùng để tìm các thông tin xuất hiện trên màn hình như bảng hiệu, caption, số áo, tên người, logo, tiêu đề bản tin.

Hướng triển khai:

- Baseline nhanh: PaddleOCR hoặc EasyOCR.
- Bản mạnh hơn theo paper: Gemini 2.0 Flash hoặc một multimodal LLM để trích text theo ngữ cảnh.
- Lưu kết quả OCR vào Elasticsearch, gắn với `frame_id` và `timestamp`.

Schema gợi ý:

```json
{
  "frame_id": "L01_V001_F001234",
  "video_id": "L01_V001",
  "timestamp": 52.36,
  "ocr_text": "Program: Financial Support",
  "language": "en"
}
```

### 3.4 ASR indexing

ASR dùng cho lời thoại, bản tin, thuyết minh, tên riêng được nói trong video.

Hướng triển khai:

- Dùng `Whisper large-v3` nếu có GPU đủ mạnh.
- Dùng `faster-whisper` với model nhỏ hơn nếu cần tốc độ.
- Segment ASR cần có timestamp start/end.
- Map mỗi ASR segment về keyframe gần nhất hoặc shot overlap tốt nhất.

Schema gợi ý:

```json
{
  "segment_id": "L01_V001_ASR_00042",
  "video_id": "L01_V001",
  "start": 51.8,
  "end": 56.1,
  "text": "the financial support program starts today",
  "nearest_frame_id": "L01_V001_F001234"
}
```

## 4. Pipeline online

### 4.1 Query decomposition

Một LLM agent nhận query gốc và trả về các sub-query theo modality:

```json
{
  "original_query": "Tìm cảnh Ronaldo ghi bàn, có tên Ronaldo trên áo",
  "expanded_queries": [
    "Ronaldo scoring a goal",
    "football player scoring a goal",
    "soccer player in a match celebrating after scoring",
    "Cristiano Ronaldo goal scene"
  ],
  "modalities": {
    "visual": {
      "query": "Ronaldo scoring a goal in a football match",
      "weight": 0.65
    },
    "ocr": {
      "query": "Ronaldo jersey name",
      "weight": 0.25
    },
    "asr": {
      "query": "Ronaldo goal",
      "weight": 0.10
    }
  }
}
```

Quy tắc quan trọng:

- Expanded query đầu tiên nên là bản dịch tiếng Anh sát nghĩa.
- Không thêm object/action không có trong query gốc.
- Visual dùng cho vật thể, hành động, màu sắc, bối cảnh.
- OCR dùng cho chữ trên màn hình.
- ASR dùng cho nội dung được nói hoặc nghe thấy.

### 4.2 Visual retrieval

Luồng visual:

1. Encode query bằng BEiT-3 và SigLIP.
2. Search Qdrant theo từng vector.
3. Gộp hai ranked list bằng Score-Reflected Reciprocal Rank Fusion hoặc một biến thể RRF có giữ similarity score.
4. Lấy top-K candidate, mặc định `K=100`.
5. Rerank bằng BLIP-2 ITM hoặc model cross-encoder tương đương.

### 4.3 Text retrieval

OCR và ASR search dùng Elasticsearch:

- Exact phrase match.
- Full-term match.
- Partial match.
- Fuzzy match.
- Boost theo field nếu query có cue rõ ràng.

Kết quả text cần map về `frame_id` hoặc `shot_id` để fusion với visual result.

### 4.4 Adaptive score fusion

Vì score từ visual, OCR và ASR khác scale, cần normalize trước:

```text
s_norm_m(f) = (s_m(f) - min(s_m)) / (max(s_m) - min(s_m) + eps)
```

Điểm fusion:

```text
S(f) = w_visual * s_visual(f)
     + w_ocr    * s_ocr(f)
     + w_asr    * s_asr(f)
```

Trong đó `w_visual`, `w_ocr`, `w_asr` đến từ query decomposition agent.

### 4.5 Temporal search

Với query gồm nhiều sự kiện, không nên chỉ trả frame có score cao nhất. Cần tạo sequence theo thứ tự thời gian.

Gợi ý:

- Tách query thành `K` event.
- Với mỗi event, lấy top `M` candidate.
- Dùng beam search với beam width `B`, paper dùng `B=8`.
- Dùng exponential decay để phạt khoảng cách thời gian quá xa.

```text
lambda_i = exp(-alpha * delta_t_i)
```

Score sequence giai đoạn exploration:

```text
SS = sum_i s_i * exp(-alpha * (t_i - t_{i-1}))
```

Final rerank có thể dùng BLIP-2 score như một gate:

```text
SS_final = sum_i s_i * lambda_i * blip2_i
```

Hyperparameter khởi đầu:

- `top_k_visual = 100`
- `query_expansion_n = 4`
- `beam_width = 8`
- `temporal_alpha = 0.01`

## 5. Cấu trúc thư mục đề xuất

```text
.
|-- README.md
|-- configs/
|   |-- default.yaml
|   |-- models.yaml
|   `-- indexes.yaml
|-- data/
|   |-- raw/
|   |-- interim/
|   |-- processed/
|   |   |-- keyframes/
|   |   |-- audio/
|   |   |-- asr/
|   |   `-- ocr/
|   `-- metadata/
|-- notebooks/
|-- scripts/
|   |-- preprocess_videos.py
|   |-- extract_keyframes.py
|   |-- build_visual_index.py
|   |-- build_text_index.py
|   `-- run_retrieval.py
|-- src/
|   |-- hcm_ai/
|   |   |-- preprocessing/
|   |   |-- embeddings/
|   |   |-- ocr/
|   |   |-- asr/
|   |   |-- indexing/
|   |   |-- retrieval/
|   |   |-- reranking/
|   |   |-- fusion/
|   |   |-- temporal/
|   |   `-- api/
|   `-- hcm_ai.egg-info/
|-- tests/
|-- docker/
|-- pyproject.toml
`-- .env.example
```

## 6. Cấu hình môi trường

Các service chính:

- Qdrant cho vector search.
- Elasticsearch cho OCR/ASR text search.
- FastAPI cho retrieval API.
- Streamlit, Gradio hoặc frontend riêng cho UI thi đấu.

Biến môi trường gợi ý:

```env
QDRANT_URL=http://localhost:6333
ELASTICSEARCH_URL=http://localhost:9200
OPENAI_API_KEY=
GOOGLE_API_KEY=
DATA_ROOT=./data
MODEL_CACHE=./models
```

## 7. API dự kiến

### Search

```http
POST /search
Content-Type: application/json
```

Request:

```json
{
  "query": "người đàn ông mặc áo đỏ đứng trước bảng hiệu ngân hàng",
  "top_k": 50,
  "mode": "auto"
}
```

Response:

```json
{
  "query_plan": {
    "visual_weight": 0.55,
    "ocr_weight": 0.35,
    "asr_weight": 0.10
  },
  "results": [
    {
      "rank": 1,
      "video_id": "L01_V001",
      "frame_id": "L01_V001_F001234",
      "timestamp": 52.36,
      "score": 0.923,
      "scores": {
        "visual": 0.81,
        "ocr": 0.97,
        "asr": 0.12,
        "rerank": 0.88
      },
      "image_path": "data/processed/keyframes/L01_V001/F001234.jpg"
    }
  ]
}
```

### Temporal Search

```http
POST /search/temporal
Content-Type: application/json
```

Request:

```json
{
  "query": "đầu tiên robot lắp khung xe, sau đó công nhân xoay tay cầm",
  "top_k": 10,
  "beam_width": 8
}
```

Response:

```json
{
  "sequences": [
    {
      "rank": 1,
      "video_id": "L02_V003",
      "score": 0.912,
      "duration": 5.1,
      "events": [
        {
          "frame_id": "L02_V003_F000120",
          "timestamp": 31.2,
          "description": "robot lắp khung xe"
        },
        {
          "frame_id": "L02_V003_F000145",
          "timestamp": 36.3,
          "description": "công nhân xoay tay cầm"
        }
      ]
    }
  ]
}
```

## 8. Roadmap triển khai

### Milestone 1 - Baseline chạy được

- Tạo metadata video.
- Extract keyframe theo interval cố định hoặc shot detection đơn giản.
- Encode keyframe bằng SigLIP hoặc CLIP.
- Index vào Qdrant.
- Search text-to-image cơ bản.
- UI hiển thị top result.

### Milestone 2 - Multimodal indexing

- Thêm OCR cho keyframe.
- Thêm ASR cho audio.
- Index OCR/ASR vào Elasticsearch.
- Map OCR/ASR result về frame/shot.
- Fusion visual + OCR + ASR bằng weight thủ công.

### Milestone 3 - Agent-guided retrieval

- Thêm query translation và query expansion.
- Thêm modality routing bằng LLM.
- Sinh weight tự động cho visual/OCR/ASR.
- Log query plan để debug.

### Milestone 4 - Reranking

- Thêm BEiT-3 + SigLIP dual retrieval.
- Gộp candidate bằng SRRF/RRF.
- Rerank top-100 bằng BLIP-2 ITM hoặc cross-encoder tương đương.
- Benchmark latency để chọn batch size phù hợp.

### Milestone 5 - Temporal reasoning

- Tách query thành event sequence.
- Lấy candidate per event.
- Beam search theo timestamp.
- Exponential decay theo khoảng cách thời gian.
- UI hiển thị sequence timeline.

### Milestone 6 - Competition hardening

- Cache mọi embedding/query expansion.
- Thêm hotkey UI cho submit nhanh.
- Thêm logging cho từng query.
- Thêm export submission.
- Viết script benchmark theo query set nội bộ.

## 9. Evaluation nội bộ

Nên tạo một bộ query nhỏ để test nhanh mỗi ngày:

```text
eval/
|-- queries_kis.jsonl
|-- queries_vqa.jsonl
|-- queries_trake.jsonl
`-- ground_truth.jsonl
```

Metric gợi ý:

- Recall@K cho KIS.
- Hit@K theo frame hoặc shot.
- MRR cho ranking.
- Latency trung bình/query.
- Với temporal: overlap theo timestamp và đúng thứ tự event.

## 10. Ghi chú thiết kế

- Ưu tiên hệ thống có thể chạy end-to-end trước khi tối ưu model.
- Không phụ thuộc vào một modality duy nhất. Nhiều query thi đấu cần OCR hoặc ASR để phân biệt các frame nhìn giống nhau.
- Reranker chỉ nên chạy trên candidate nhỏ vì chi phí cao.
- Cần lưu đầy đủ intermediate artifact để không phải preprocess lại toàn bộ dataset.
- Query tiếng Việt nên được dịch/expand sang tiếng Anh cho nhánh visual embedding, nhưng vẫn giữ tiếng Việt cho OCR/ASR nếu dữ liệu có tiếng Việt.
- UI thi đấu phải ưu tiên tốc độ thao tác: search nhanh, preview rõ timestamp, copy/submit result nhanh.

## 11. Tài liệu tham khảo

- Paper nền tảng: https://arxiv.org/pdf/2512.12935
- Qdrant: https://qdrant.tech/documentation/
- Elasticsearch: https://www.elastic.co/guide/
- Whisper: https://github.com/openai/whisper
- faster-whisper: https://github.com/SYSTRAN/faster-whisper
- TransNetV2: https://github.com/soCzech/TransNetV2
