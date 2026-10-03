# Financial Sentiment Encoder (FSE) & Market Liquidity Index System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-green.svg)
![License](https://img.shields.io/badge/License-Academic-lightgrey.svg)

Hệ thống định lượng hóa **Chỉ số Cảm xúc Dòng tiền Thị trường ($S_t$)** từ tin tức tài chính tiếng Việt thuộc ngành Ngân hàng (27 mã cổ phiếu niêm yết). Hệ thống tích hợp mô hình học sâu **FSE (Financial Sentiment Encoder)** dựa trên PhoBERT-base, cơ chế Tri thức Đa Giáo viên (Multi-Teacher Knowledge Distillation), tìm kiếm vector tương đồng (FAISS) và các lý thuyết tài chính hành vi (Prospect Theory, Memory Decay, Intraday Time Decay).

---

## Mục lục

1. [Tổng quan Hệ thống](#-tổng-quan-hệ-thống)
2. [Kiến trúc Mô hình & Phương pháp luận](#-kiến-trúc-mô-hình--phương-pháp-luận)
3. [Cấu trúc Thư mục Dự án](#-cấu-trúc-thư-mục-dự-án)
4. [Hướng dẫn Cài đặt & Môi trường](#-hướng-dẫn-cài-đặt--môi-trường)
5. [Cấu hình Hệ thống (Configuration)](#-cấu-hình-hệ-thống-configuration)
6. [Quy trình Vận hành Pipeline](#-quy-trình-vận-hành-pipeline)
7. [Tối ưu hóa Tham số Kinh tế lượng ($\Theta$)](#-tối-ưu-hóa-tham-số-kinh-tế-lượng-)
8. [Metrics Đánh giá & Kiểm định](#-metrics-đánh-giá--kiểm-định)
9. [Lộ trình Phát triển (Roadmap)](#-lộ-trình-phát-triển-roadmap)

---

## Tổng quan Hệ thống

Chỉ số $S_t \in [-1, 1]$ đóng vai trò là một biến số kinh tế lượng đầu vào, phản ánh trạng thái tâm lý, niềm tin và xu hướng hành vi của lực lượng **nhà đầu tư cá nhân (Retail Investors)** trên thị trường chứng khoán Việt Nam.

* **Phạm vi dữ liệu**: 04 nguồn tin chính thống (Thời báo Tài chính Việt Nam, Vietnam Finance, CafeF, Vietstock).
* **Phạm vi ngành**: 27 mã cổ phiếu Ngân hàng niêm yết (20 HOSE, 2 HNX, 5 UPCoM).
* **Ứng dụng**: Mô hình VAR, EGARCH-X, kiểm định nhân quả Granger Causality đối với biến động giá ($R_t$) và thanh khoản thị trường.

---

## Kiến trúc Mô hình & Phương pháp luận

### 1. Kiến trúc FSE (Financial Sentiment Encoder)
Mô hình FSE xử lý văn bản thô theo 03 tầng:
* **Tầng 1 (Backbone Encoder)**: PhoBERT-base (`VinAI/phobert-base-v2`) kết hợp tách từ hình thái học tiếng Việt (`VnCoreNLP` / `pyvi`).
* **Tầng 2 (Attention Pooling Layer)**: Chia bài báo thành các đoạn $S_1, S_2, \dots, S_m$, trích xuất vector $h_k \in \mathbb{R}^{768}$ và nén lại bằng cơ chế Attention Weighting ($\alpha_k$) để thu được Vector Đại diện Bài báo $V_{\text{article}} \in \mathbb{R}^{768}$.
* **Tầng 3 (Multi-Task Classification Heads)**:
  1. *Sentiment Regression Head*: Xuất điểm cảm xúc ròng $NSS_j \in [-1, 1]$ qua hàm $\tanh$.
  2. *Relevance Tier Head (CORAL Ordinal Regression)*: Dự đoán phân tầng mức độ liên quan $Tier_j \in \{1, 2, 3, 4\}$.

### 2. Qúa trình Huấn luyện: DAPT & Multi-Teacher Distillation
* **Domain-Adaptive Pre-training (DAPT)**: Masked Language Modeling (MLM) trên corpus tin tức tài chính tiếng Việt.
* **Multi-Teacher Ensemble**: Sử dụng GPT-4o, Claude 3.5 Sonnet và Gemini 1.5 Pro gán nhãn cho 20,000 bài báo.
* **Consensus Filtering**: Chỉ giữ lại các mẫu có độ lệch chuẩn $\sigma \le 0.25$ và thống nhất về dấu cảm xúc.
* **Knowledge Distillation**: Huấn luyện FSE bằng hàm mất mát tổng hợp:
  $$L_{\text{total}} = \alpha \cdot L_{\text{MSE}} + \beta \cdot L_{\text{CE}} + \lambda_{\text{KD}} \cdot L_{\text{KD}}$$

### 3. Không gian Vector, Novelty & Decay
* **Độ trùng lặp (Redundancy - $Sim_j$)**: Độ tương đồng Cosine cực đại trong cửa sổ 48 giờ quá khứ sử dụng FAISS Vector DB.
* **Novelty Score ($N_j$)**:
  $$N_j = \begin{cases} 0 & \text{nếu } Sim_j \ge 0.85 \\ \exp\left(-\kappa \cdot \frac{Sim_j - 0.3}{0.55}\right) & \text{nếu } 0.3 \le Sim_j < 0.85 \\ 1 & \text{nếu } Sim_j < 0.3 \end{cases}$$

### 4. Quy trình Tính Chỉ số Cảm xúc Ngày ($S_t$)
1. **Trọng số Bài báo ($W_j$)**: $W_j = PositionWeight_j \times SourceCredibility_j$ (xác định qua P1-P6 vị trí hiển thị DOM/API và phương pháp MCDM Entropy Weight Method + Delphi).
2. **Điểm Cảm xúc Thô ($S_t^{\text{raw}}$)**:
  $$S_t^{\text{raw}} = \frac{\sum_{j \in N_t} NSS_j \cdot Tier_j \cdot W_j \cdot N_j \cdot \exp(-\mu \cdot \Delta t_j)}{\sum_{j \in N_t} Tier_j \cdot W_j \cdot N_j \cdot \exp(-\mu \cdot \Delta t_j)}$$
3. **Phản ứng Bất đối xứng (Prospect Theory)**: $\Phi(S_t^{\text{raw}}) = S_t^{\text{raw}}$ nếu $S_t^{\text{raw}} \ge 0$ và $\gamma \cdot S_t^{\text{raw}}$ nếu $S_t^{\text{raw}} < 0$.
4. **Trễ Tâm lý (Memory Decay)**: $S_t = \lambda \cdot S_{t-1} + (1 - \lambda) \cdot \Phi(S_t^{\text{raw}})$.

---

## Cấu trúc Thư mục Dự án

```text
fse-sentiment-index/
├── config/                         # Quản lý cấu hình & siêu tham số
│   ├── settings.yaml               # Settings chung (Paths, Crawler, FAISS, DB)
│   ├── hyperparameters.yaml        # Siêu tham số PhoBERT, AdamW, Loss Weights
│   └── quant_params.yaml           # Siêu tham số Vector Theta (gamma, lambda, mu, alpha, W)
├── data/                           # Lưu trữ dữ liệu các giai đoạn
│   ├── raw/                        # HTML thô
│   ├── parsed/                     # JSON bài báo đã parse
│   ├── processed/                  # Dữ liệu chuẩn hóa ngôn ngữ (VnCoreNLP)
│   ├── gold_standard/              # Tập 300 bài báo kiểm định chuẩn (Human-annotated)
│   └── embeddings/                 # FAISS vector index files
├── src/                            # Source code chính
│   ├── crawler/                    # Discovery, Fetcher (Playwright), Parser, Dedup
│   ├── models/                     # Backbone, Attention Pooling, Heads, FSE, Distillation
│   ├── storage/                    # SQLAlchemy models & FAISS Vector Store wrapper
│   ├── index_engine/               # Novelty, Weights, Intraday Decay, Behavioral, Aggregator
│   ├── optimization/               # MCDM Entropy Weight, Objective, Sequential Optimizer
│   └── utils/                      # Vietnamese NLP utils, Trading Calendar, Logger
├── notebooks/                      # Jupyter Notebooks nghiên cứu & kiểm định kinh tế lượng
├── tests/                          # Unit tests & Integration tests
├── scripts/                        # Automation scripts (Daily run, Training, Optimization)
├── constants.py                    # Danh sách 27 mã Ngân hàng, Sàn, Baseline values
├── requirements.txt                # Thư viện phụ thuộc
├── Dockerfile                      # Containerization setup
└── README.md                       # Tài liệu hướng dẫn sử dụng
```

---

## Hướng dẫn Cài đặt & Môi trường

### 1. Yêu cầu Hệ thống
* **OS**: Linux (Ubuntu 20.04/22.04 khuyến nghị) hoặc macOS.
* **Python**: `3.10+`
* **GPU**: NVIDIA GPU với VRAM $\ge$ 16GB (NVIDIA T4 / A100) hỗ trợ CUDA 11.8+.

### 2. Cài đặt Môi trường
```bash
# Clone repository
git clone https://github.com/your-org/fse-sentiment-index.git
cd fse-sentiment-index

# Tạo và kích hoạt môi trường ảo Python
python3 -m venv venv
source venv/bin/activate

# Cập nhật pip và cài đặt thư viện phụ thuộc
pip install --upgrade pip
pip install -r requirements.txt

# Cài đặt Playwright Browsers (cho Crawler)
playwright install chromium
```

---

## Cấu hình Hệ thống (Configuration)

Các tham số hệ thống được quản lý tập trung trong thư mục `config/`:

* **`config/settings.yaml`**: Cấu hình kết nối Database, lưu trữ FAISS vector index, rate limiting cho crawler.
* **`config/hyperparameters.yaml`**: Tham số huấn luyện PyTorch/Transformers (Learning rate, Batch size, Weight decay, $\alpha, \beta, \lambda_{\text{KD}}$).
* **`config/quant_params.yaml`**: Tham số vector $\Theta = (\gamma, \lambda, \mu, \alpha, W_{\text{source}}, W_{\text{tier}})$.

---

## Quy trình Vận hành Pipeline

### 1. Thu thập & Tiền xử lý Dữ liệu
Chạy luồng crawler tự động phát hiện, tải, trích xuất HTML, phân tầng vị trí (P1-P6) và tách từ tiếng Việt:
```bash
python -m src.crawler.pipeline --sources cafef,vietstock,vneconomy,vietnamfinance
```

### 2. Huấn luyện Mô hình FSE
Thực hiện quá trình Fine-tuning FSE với Distillation từ Multi-Teacher Ensemble:
```bash
python scripts/train_fse.py --config config/hyperparameters.yaml
```

### 3. Chạy Pipeline Tính toán Chỉ số $S_t$ Hàng ngày
Tự động chạy tác vụ thu thập, suy luận FSE, tính Novelty Score qua FAISS, và tổng hợp ra chỉ số $S_t$ cho phiên giao dịch:
```bash
python scripts/run_daily_pipeline.py --date 2026-10-02
```

---

## Tối ưu hóa Tham số Kinh tế lượng ($\Theta$)

Toàn bộ tham số $\Theta = (\gamma, \lambda, \mu, \alpha, W_{\text{source}}, W_{\text{tier}})$ được ước lượng tối ưu hóa liên tục từ dữ liệu thực tế nhằm tối đa hóa tương quan với tỷ suất sinh lời bất thường (Abnormal Return - $R_t$):

$$\Theta^* = \arg\max_{\Theta} \text{Corr}(S_t(\Theta), R_t)$$

Chạy quy trình Sequential Optimization (Tối ưu hóa tuần tự) sử dụng Optuna / Walk-Forward Validation:
```bash
python scripts/optimize_theta.py --method sequential --trials 500
```

---

## Metrics Đánh giá & Kiểm định

Mô hình và Chỉ số $S_t$ bắt buộc phải thỏa mãn các tiêu chí kiểm định nghiêm ngặt:

| Bài toán / Kiểm định | Metrics / Test | Ngưỡng Mục tiêu |
| :--- | :--- | :--- |
| **Sentiment Regression** | Pearson Correlation / MAE | $\ge 0.70$ / $\le 0.15$ |
| **Tier Classification** | F1-macro / MAE (ordinal) | $\ge 0.75$ / $\le 0.30$ |
| **Knowledge Distillation** | KL Divergence | $\le 0.10$ |
| **Tính dừng (Stationarity)** | ADF Test / PP Test | $p \text{-value} \le 0.05$ |
| **Quan hệ Nhân quả** | Granger Causality ($S_t \to R_t$) | $p \text{-value} \le 0.05$ |
| **Tác động Biến động** | EGARCH-X / VAR | R² $\ge 0.30$ |

Đánh giá mô hình trên tập Gold Standard (300 bài báo kiểm định thủ công):
```bash
python -m tests.evaluate_gold_standard
```

---

## 🗺 Lộ trình Phát triển (Roadmap)

* [x] **Giai đoạn 1**: Xây dựng Corpus chuyên biệt 27 Ngân hàng, hoàn thiện Crawler Playwright & Baseline MCDM cho SourceCredibility.
* [x] **Giai đoạn 2**: Đóng băng trọng số FSE v1.0, tích hợp Attention Pooling và Distillation từ Ensemble GPT-4o / Claude 3.5 / Gemini 1.5 Pro.
* [ ] **Giai đoạn 3**: Nâng cấp FSE lên **Aspect-Based Sentiment Analysis (ABSA)**, bóc tách điểm cảm xúc ròng theo từng mã Ticker riêng biệt trong cùng một bài báo.
* [ ] **Giai đoạn 4**: Tích hợp Real-time Optimization Engine và đẩy dữ liệu $S_t$ tự động vào hệ thống giao dịch thuật toán (Algorithmic Trading System).

---

## Bản quyền & Cảnh báo Nghiên cứu

Project này được phát triển phục vụ mục đích **Nghiên cứu Học thuật & Kinh tế lượng Định lượng**. 
* Tự động hóa crawl tuân thủ quy định `robots.txt`, giới hạn rate limit 1 request/2s/domain và không gây quá tải server nguồn.
* Dữ liệu và chỉ số xuất ra không phải là lời khuyên đầu tư tài chính trực tiếp.