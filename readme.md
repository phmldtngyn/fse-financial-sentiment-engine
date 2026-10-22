# FSE Engine — Financial Sentiment & Banking Correlation Analysis

Hệ thống phân tích cảm xúc tin tức tài chính tiếng Việt và khảo sát mối liên hệ giữa sentiment với lợi nhuận, thanh khoản và biến động của nhóm cổ phiếu ngân hàng Việt Nam.

> **Kết quả được báo cáo:** 8/8 cặp tương quan có ý nghĩa thống kê ở mức `p < 0.05` trên 2.492 ngày giao dịch trong giai đoạn 2016–2026. Sentiment tích cực có tương quan dương với return và tương quan âm với intraday range; độ lớn sentiment `|S_t|` có tương quan dương với thanh khoản.

> **Lưu ý:** Các giá trị `F1-macro = 0.XX` bên dưới là chỗ giữ chỗ, cần thay bằng kết quả đánh giá thực tế trước khi công bố. Tương quan thống kê không tự chứng minh quan hệ nhân quả. Hãy xác minh lại số liệu, dữ liệu đầu vào và khả năng tái lập trước khi sử dụng các kết quả trong báo cáo học thuật.

---

## Mục lục

- [Tổng quan](#tổng-quan)
- [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
- [Dataset](#dataset)
- [Model](#model)
- [Kết quả phân tích tương quan](#kết-quả-phân-tích-tương-quan)
- [Cài đặt](#cài-đặt)
- [Sử dụng](#sử-dụng)
- [API endpoints](#api-endpoints)
- [Dashboard](#dashboard)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Reproducibility](#reproducibility)
- [Hạn chế và hướng mở rộng](#hạn-chế-và-hướng-mở-rộng)
- [Tác giả](#tác-giả)
- [License](#license)

---

## Tổng quan

### Câu hỏi nghiên cứu

> **Cảm xúc tin tức tài chính (`S_t`) có tương quan với thanh khoản, lợi nhuận và biến động của nhóm cổ phiếu ngân hàng Việt Nam hay không?**

### Các thành phần chính

1. **Mô hình sentiment 3 lớp:** fine-tune PhoBERT-base-v2 để phân loại tin tức thành tiêu cực, trung tính và tích cực.
2. **Phân tích tương quan:** khảo sát mối liên hệ giữa `S_t` và các chỉ số thị trường gồm volume, return và intraday range.
3. **Pipeline dữ liệu:** chuẩn bị dữ liệu, ghép sentiment với dữ liệu ngân hàng, phân tích thống kê và tạo biểu đồ.
4. **Ứng dụng:** cung cấp API bằng FastAPI và giao diện phân tích bằng Streamlit; dự án có cấu hình Docker.

### Các kết quả được báo cáo

- Bộ dữ liệu huấn luyện sentiment gồm **6.458 câu**.
- Bộ dữ liệu tương quan gồm **2.492 ngày** từ `2016-04-22` đến `2026-04-14`.
- **8/8 cặp tương quan** trong bảng kết quả có `p < 0.05`.
- `|S_t|` có tương quan với `log(Volume)` cao hơn tương quan giữa `S_t` và volume trong các kết quả được liệt kê.

---

## Kiến trúc hệ thống

```text
                    FSE ENGINE PIPELINE

  News sources                         VN-Quant dataset
  CafeF, VnExpress, ...                6,458 LLM-labeled sentences
          |                                      |
          v                                      v
  +----------------+                    +-------------------+
  | Collector      |                    | Preprocessing     |
  | APScheduler    |                    | Dataset join      |
  | 30-minute job  |                    +---------+---------+
  +--------+-------+                              |
           |                                      v
           |                            +-------------------+
           |                            | SFT training      |
           |                            | PhoBERT 3-class   |
           |                            +---------+---------+
           |                                      |
           |                                      v
           |                            +-------------------+
           |                            | Model checkpoint  |
           |                            | best.pt           |
           |                            +---------+---------+
           |                                      |
           +------------------+-------------------+
                              |
                              v
                    +---------------------+
                    | FastAPI + Streamlit |
                    | API and dashboard   |
                    +---------------------+
```

The correlation-analysis pipeline is run separately through scripts that extract daily banking data, join it with sentiment, and calculate statistical associations.

---

## Dataset

### 1. Dataset huấn luyện sentiment

| Thuộc tính | Giá trị |
|---|---|
| Nguồn | VN-Quant: kết hợp `sentiment_LLM_labeled` và `google_news_10years_FULLTEXT` |
| Số câu | 6.458 |
| Phân bố nhãn | 12,1% negative / 66,3% neutral / 21,6% positive |
| Chia tập | 5.166 train / 646 validation / 646 test |
| Độ dài văn bản | 50–800 ký tự; cắt theo giới hạn đầu vào của PhoBERT |
| Nhãn | 3 lớp được tạo từ `Sentiment_Score` trong `[-1, 1]`, dùng ngưỡng `±0.3` |
| Định dạng được báo cáo | `data/processed/*_3class.parquet` |

Các file train, validation và test được mô tả là đã commit trong repository. Kiểm tra thư mục `data/processed/` để xác nhận chúng có mặt trong phiên bản hiện tại.

### 2. Dataset phân tích tương quan

| Thuộc tính | Giá trị |
|---|---|
| Nguồn sentiment `S_t` | VN-Quant `Sentiment_NLP` (Gemini LLM-labeled) |
| Nguồn dữ liệu ngân hàng | VN-Quant `UNIFIED_RESEARCH_MATRIX` |
| Số mã ngân hàng | 26 |
| Số ngày | 2.492 (`2016-04-22` đến `2026-04-14`) |
| Tỷ lệ ngày có `S_t` khác 0 | 66,7% theo thống kê được cung cấp |
| Mã | VCB, BID, CTG, TCB, VPB, MBB, ACB, STB, SHB, HDB, LPB, TPB, VIB, MSB, OCB, SSB, EIB, NAB, BVB, SGB, VBB, KLB, PGB, ABB, BAB, PVCB |

---

## Model

### Kiến trúc FSE

```text
Input text (maximum 256 tokens)
          |
          v
+--------------------------+
| PhoBERT-base-v2          | 135M parameters, 12 layers
| Backbone encoder         |
+------------+-------------+
             |
             v
      [CLS] embedding (768d)
             |
             v
+--------------------------+
| Linear classification   | 768 -> 3
| head                     |
+------------+-------------+
             |
             v
      logits -> softmax
             |
             v
 negative / neutral / positive
```

### Cấu hình huấn luyện

| Tham số | Giá trị được cung cấp |
|---|---|
| Backbone | `VinAI/phobert-base-v2` |
| Loss | CrossEntropy với class weights `[2.75, 0.50, 1.54]` |
| Optimizer | AdamW |
| Learning rate | `2e-5` |
| Weight decay | `0.01` |
| Batch size | `32` |
| Epochs | `5` |
| Max length | `256` tokens |
| Môi trường huấn luyện | Tesla T4 (Google Colab) |
| Best test F1-macro | **`0.XX` — cần cập nhật bằng kết quả thực tế** |

### Checkpoint

- Đường dẫn dự kiến: `checkpoints/fse_sft/best.pt`.
- Checkpoint được mô tả có kích thước khoảng 500 MB và không được commit lên GitHub do giới hạn kích thước tệp.
- Có thể huấn luyện lại theo [`docs/training.md`](docs/training.md), nếu file hướng dẫn này có trong repository, hoặc lấy checkpoint từ nơi lưu trữ được dự án chỉ định.
- Đặt checkpoint tại `checkpoints/fse_sft/best.pt` trước khi chạy các thành phần cần mô hình đã huấn luyện.

---

## Kết quả phân tích tương quan

### Bảng tương quan

`n = 2.492`, giai đoạn 2016–2026. Các số liệu dưới đây là kết quả được cung cấp trong bản thảo README và cần được đối chiếu với file kết quả phân tích trước khi công bố.

| Cặp tương quan | Pearson r | p-value | Spearman ρ | Mức ý nghĩa |
|---|---:|---:|---:|---|
| `S_t` ↔ Return | **+0.1414** | `<0.001` | +0.1134 | `***` |
| `S_t` ↔ Intraday Range | **−0.1223** | `<0.001` | −0.0756 | `***` |
| `|S_t|` ↔ `log(Volume)` | **+0.1595** | `<0.001` | +0.1607 | `***` |
| `S_t` ↔ `|Return|` | −0.0770 | `<0.001` | −0.0544 | `***` |
| `S_t` ↔ Volume | +0.0612 | 0.002 | +0.0577 | `**` |
| `|S_t|` ↔ Intraday Range | +0.0519 | 0.010 | +0.0264 | `*` |
| `|S_t|` ↔ `|Return|` | +0.0514 | 0.010 | +0.0203 | `*` |
| `S_t` ↔ `log(Volume)` | +0.0494 | 0.014 | +0.0577 | `*` |

Quy ước thường dùng: `*** p < 0.001`, `** p < 0.01`, `* p < 0.05`. Cần xác nhận quy ước này khớp với mã phân tích. Theo bảng trên, cả 8 cặp đều có `p < 0.05`; ý nghĩa thống kê không đồng nghĩa với tác động lớn về mặt kinh tế.

### Diễn giải thận trọng

#### 1. Sentiment và return / intraday range

- `S_t` và Return: `r = +0.1414`, tương quan dương ở mức nhỏ.
- `S_t` và Intraday Range: `r = −0.1223`, tương quan âm ở mức nhỏ.

Các kết quả này cho thấy sentiment tích cực đi cùng return cao hơn và intraday range thấp hơn trong mẫu được phân tích. Chúng **không tự chứng minh** rằng sentiment gây ra biến động giá hoặc thị trường ổn định hơn. Cần kiểm soát các yếu tố khác và kiểm tra độ bền theo thời gian.

#### 2. Độ lớn sentiment và thanh khoản

- `S_t` ↔ Volume: `r = +0.0612`.
- `|S_t|` ↔ `log(Volume)`: `r = +0.1595`.

Trong bảng kết quả, độ lớn sentiment có tương quan dương lớn hơn với log-volume so với tương quan giữa sentiment có dấu và volume. Điều này tương thích với giả thuyết rằng mức độ nổi bật của thông tin có thể liên quan đến hoạt động giao dịch, nhưng cần kiểm định bổ sung để xác nhận cơ chế.

#### 3. So sánh với nghiên cứu trước

Không nên kết luận effect size của dự án “vượt literature” nếu chưa đối chiếu các nghiên cứu trên cùng định nghĩa biến, tần suất, thị trường, thiết kế mẫu và phương pháp ước lượng. Các hệ số tương quan giữa những nghiên cứu khác nhau không nhất thiết so sánh trực tiếp được.

### Báo cáo và biểu đồ

Báo cáo chi tiết dự kiến ở [`docs/correlation_report.md`](docs/correlation_report.md). Các biểu đồ đầu ra dự kiến nằm trong `data/processed/correlation/`:

- `scatter_S_t_vs_Return.png`
- `scatter_S_t_vs_Range.png`
- `scatter_S_t_vs_logVolume.png`
- `scatter_abs_S_t_vs_Range.png`
- `rolling_correlation.png` — tương quan trượt cửa sổ 60 ngày

---

## Cài đặt

### Yêu cầu

- Python 3.10 trở lên
- Git
- RAM 8 GB (khuyến nghị 16 GB)
- Kết nối Internet khi tải các dependency hoặc model cần thiết

### Cài đặt local trên Windows

```powershell
# 1. Clone repository
git clone https://github.com/phmldtngyn/fse-financial-sentiment-engine.git
cd fse-financial-sentiment-engine

# 2. Tạo và kích hoạt virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Cài dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Đặt checkpoint tại checkpoints/fse_sft/best.pt
# nếu muốn chạy các chức năng cần mô hình đã huấn luyện.

# 5. Kiểm tra import (nếu module có trong phiên bản repository hiện tại)
python -c "from src.models.fse_model import FSEModel; print('OK')"
```

Trên Linux/macOS, kích hoạt môi trường bằng `source venv/bin/activate`.

### Dependencies chính

Các gói chính được dự án khai báo hoặc sử dụng theo bản mô tả hiện tại:

```text
torch>=2.0.0
transformers>=4.30.0
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
streamlit>=1.40.0
sentence-transformers>=3.0.0
faiss-cpu>=1.8.0
pandas>=2.0.0
scikit-learn>=1.3.0
scipy>=1.11.0
matplotlib>=3.8.0
statsmodels>=0.14.0
```

`requirements.txt` trong repository là nguồn tham chiếu cuối cùng cho các phiên bản dependency thực tế.

---

## Sử dụng

Các lệnh dưới đây phản ánh cấu trúc dự án dự kiến. Nếu một script hoặc module chưa có trong nhánh hiện tại, hãy kiểm tra tên và đường dẫn thực tế trong repository.

### 1. Huấn luyện mô hình 3 lớp

```bash
python -m src.training.train_fse_sft
```

Đầu ra dự kiến: `checkpoints/fse_sft/best.pt` và các chỉ số đánh giá trên validation set.

### 2. Chạy crawler

```bash
# Chạy một lần
python -m scripts.run_collector

# Chạy nền theo chu kỳ 30 phút
python -m scripts.run_collector --daemon --interval 30
```

Đầu ra dự kiến: `data/parsed/crawled_data_*.json`.

### 3. Phân tích tương quan

```bash
# Bước 1: trích xuất dữ liệu ngân hàng theo ngày (26 mã)
python -m scripts.extract_banking_daily

# Bước 2: ghép sentiment với dữ liệu ngân hàng
python -m scripts.join_banking_sentiment

# Bước 3: tính tương quan và tạo biểu đồ
python -m scripts.correlation_analysis
```

Đầu ra dự kiến:

- `data/processed/correlation/correlation_results.csv`
- `data/processed/correlation/correlation_lag.csv`
- Các biểu đồ `.png` trong `data/processed/correlation/`

### 4. Chạy API và dashboard

Terminal 1 — API:

```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Terminal 2 — dashboard:

```bash
streamlit run dashboard/app.py
```

Sau khi khởi động thành công:

- API docs: <http://localhost:8000/docs>
- Dashboard: <http://localhost:8501>

---

## API endpoints

Base URL mặc định: `http://localhost:8000`.

### `POST /api/v1/sentiment`

Chấm điểm cảm xúc văn bản theo mô hình 3 lớp.

```bash
curl -X POST http://localhost:8000/api/v1/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text":"Cổ phiếu VNM tăng trần, khối ngoại mua ròng"}'
```

Ví dụ response được mô tả:

```json
{"score": 0.7, "tier": 3}
```

Quy ước score dự kiến: negative `−0.7`, neutral `0.0`, positive `+0.7`. Hãy đối chiếu với schema và code endpoint để xác nhận response thực tế.

### `POST /api/v1/rag/query`

Truy vấn ngữ cảnh RAG với cơ chế rerank có xét thời gian.

```bash
curl -X POST http://localhost:8000/api/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query":"cổ phiếu ngân hàng tăng","top_m":5,"window_hours":720}'
```

### `GET /api/v1/correlation`

Lấy kết quả phân tích tương quan.

```bash
curl http://localhost:8000/api/v1/correlation
```

Response dự kiến là một mảng các object, gồm các trường như `pair`, `pearson_r`, `pearson_p` và `significant`.

### `GET /health`

Health check cho ứng dụng và các thành phần đã nạp.

Ví dụ response dự kiến:

```json
{
  "status": "ok",
  "models_loaded": {
    "sentiment": true,
    "rag": true,
    "correlation": true
  }
}
```

Các response JSON trên chỉ là ví dụ minh họa; response thực tế phụ thuộc vào implementation trong repository.

---

## Dashboard

Dashboard Streamlit được mô tả gồm ba tab:

### Sentiment

- Nhập văn bản tin tức để phân loại sentiment 3 lớp.
- Hiển thị score, tier và diễn giải nếu được hỗ trợ bởi implementation.

### RAG

- Truy vấn ngữ cảnh từ FAISS index.
- Hiển thị các kết quả top-K, similarity và thông tin thời gian nếu có.

### Banking Correlation

- Hiển thị các cặp tương quan với Pearson, Spearman và p-value.
- Trực quan hóa scatter plots và rolling correlation.
- Cung cấp diễn giải kết quả thống kê; các diễn giải lý thuyết cần được xem là giả thuyết cho đến khi có kiểm định phù hợp.

---

## Cấu trúc dự án

```text
fse-financial-sentiment-engine/
├── api/                         # FastAPI backend
│   ├── main.py                  # App entry point / lifespan
│   ├── schemas.py               # Pydantic schemas
│   └── routes/
│       ├── sentiment.py         # Sentiment endpoint
│       ├── rag.py               # RAG query endpoint
│       └── correlation.py       # Correlation endpoint
├── dashboard/
│   └── app.py                   # Streamlit dashboard
├── src/
│   ├── crawler/                 # News collector pipeline
│   │   ├── discovery.py         # Feed discovery
│   │   ├── fetcher.py           # Page fetching / Playwright
│   │   ├── parser.py            # HTML to text
│   │   ├── dedup.py             # Deduplication
│   │   ├── pipeline.py          # Pipeline orchestration
│   │   └── scheduler.py         # APScheduler
│   ├── data/
│   │   └── dataset.py            # Dataset classes
│   ├── models/
│   │   └── fse_model.py         # PhoBERT + 3-class head
│   ├── training/
│   │   ├── losses.py            # Training losses
│   │   └── train_fse_sft.py     # Training entry point
│   ├── rag/                     # RAG components
│   │   ├── embedder.py          # Sentence embeddings
│   │   ├── indexer.py           # FAISS index
│   │   └── retriever.py         # Retrieval / time weighting
│   └── storage/
│       └── vector_db.py         # Vector storage
├── scripts/                     # CLI scripts
│   ├── run_collector.py
│   ├── build_rag_index.py
│   ├── prepare_vn_quant_dataset.py
│   ├── prepare_3class_dataset.py
│   ├── extract_banking_daily.py
│   ├── join_banking_sentiment.py
│   └── correlation_analysis.py
├── config/
│   ├── hyperparameters.yaml
│   └── rag.yaml
├── data/
│   ├── processed/               # Prepared datasets and outputs
│   │   └── correlation/
│   ├── parsed/                  # Crawler output (may be ignored by Git)
│   └── raw/                     # Raw datasets (may be ignored by Git)
├── docs/
│   └── correlation_report.md
├── archive/                     # Archived experiments
│   └── forecasting/
├── checkpoints/                 # Model weights (typically not committed)
├── Dockerfile.api
├── Dockerfile.dashboard
├── docker-compose.yml
├── requirements.txt
└── README.md
```

Đây là sơ đồ cấu trúc theo mô tả hiện tại, không thay thế việc kiểm tra các file thực sự có trong repository.

---

## Reproducibility

Để người khác có thể tái lập kết quả, repository nên lưu trữ hoặc mô tả rõ các thành phần sau.

### Dữ liệu và kết quả

Các file được báo cáo là đã commit:

- `data/processed/train_3class.parquet`
- `data/processed/val_3class.parquet`
- `data/processed/test_3class.parquet`
- `data/processed/banking_daily.parquet`
- `data/processed/banking_with_sentiment.parquet`
- Các file `.csv` và `.png` trong `data/processed/correlation/`

### Cấu hình và random seed

- `config/hyperparameters.yaml` — cấu hình huấn luyện.
- `config/rag.yaml` — cấu hình RAG.
- Seed được báo cáo: `random_state=42` cho các bước chia dữ liệu và `random.seed(42)` cho sampling, nếu áp dụng trong code tương ứng.

### Kiểm định theo thời gian

Bản mô tả dự án nêu walk-forward CV với cửa sổ rolling `180/30/30` ngày và nguyên tắc train trước, test sau. Cần xác nhận quy trình này thực sự được thực thi trong mã phân tích và không có leakage từ nhãn, chuẩn hóa, chọn biến hoặc bước tiền xử lý.

**Không nên mô tả pipeline là “tái lập 100%” nếu chưa kiểm tra độc lập** khả năng tải đủ dữ liệu, phiên bản dependency, checkpoint, seed, lệnh chạy và việc tái tạo các bảng/biểu đồ.

---

## Hạn chế và hướng mở rộng

### Hạn chế

1. **Nhãn sentiment từ LLM:** nhãn do Gemini tạo có thể chứa noise và không tương đương với nhãn chuyên gia.
2. **Tương quan không phải nhân quả:** cần mô hình và thiết kế kiểm định bổ sung để đánh giá chiều tác động.
3. **Phạm vi ngành:** kết quả từ nhóm ngân hàng không mặc nhiên khái quát cho toàn thị trường Việt Nam.
4. **Dữ liệu ngày:** tổng hợp theo ngày có thể bỏ qua động lực intraday.
5. **Effect size:** hệ số có ý nghĩa thống kê vẫn có thể có độ lớn kinh tế nhỏ; nên báo cáo khoảng tin cậy và kiểm tra độ bền.
6. **Nhiều phép kiểm định:** cân nhắc hiệu chỉnh multiple testing khi diễn giải đồng thời nhiều cặp tương quan.

### Hướng mở rộng

1. **Granger causality tests:** khảo sát khả năng dự báo theo thời gian giữa sentiment và các chỉ số ngân hàng.
2. **EGARCH-X:** kiểm tra sentiment như biến ngoại sinh trong mô hình volatility.
3. **Sentiment từ FSE:** tạo chuỗi `S_t` bằng mô hình PhoBERT sau khi xác nhận chất lượng trên tập test độc lập và corpus lịch sử phù hợp.
4. **Biến vĩ mô và dòng vốn:** bổ sung các biến như VIX, DXY, dòng vốn nước ngoài và dư nợ margin khi dữ liệu phù hợp.
5. **Event study:** nghiên cứu phản ứng quanh các sự kiện như quyết định Fed, NHNN và công bố kết quả kinh doanh.
6. **Mở rộng ngành:** thử nghiệm trên bất động sản, chứng khoán và sản xuất.

---

## Tác giả

**Phạm Lê Đạt Nguyên**

- Repository: [phmldtngyn/fse-financial-sentiment-engine](https://github.com/phmldtngyn/fse-financial-sentiment-engine)
- Báo cáo tương quan: [`docs/correlation_report.md`](docs/correlation_report.md)

---

## License

Đồ án nghiên cứu, dự kiến sử dụng cho mục đích học thuật. Nếu phân phối công khai, hãy bổ sung một license cụ thể và kiểm tra quyền sử dụng đối với dữ liệu, mô hình nền và các nguồn tin được thu thập.
