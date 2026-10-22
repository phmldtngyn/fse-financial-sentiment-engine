\# Section 4: Kết quả phân tích tương quan



\## 4.1. Phương pháp



\- \*\*Dữ liệu\*\*: 2,492 ngày giao dịch (2016-04-22 → 2026-04-14)

\- \*\*S\_t\*\*: Sentiment\_NLP từ VN-Quant (aggregate daily average, LLM-labeled Gemini)

\- \*\*Banking indicators\*\* (26 mã ngân hàng niêm yết):

&#x20; - `total\_volume`: tổng khối lượng giao dịch

&#x20; - `avg\_return`: lợi nhuận trung bình

&#x20; - `intraday\_range`: biên độ dao động `(high-low)/close` trung bình

\- \*\*Phương pháp\*\*: Pearson + Spearman correlation với p-value, lag analysis (0-5 ngày)



\## 4.2. Kết quả chính



| Cặp tương quan | Pearson r | p-value | Spearman ρ | Ý nghĩa |

|----------------|-----------|---------|-----------|---------|

| S\_t ↔ Return | \*\*+0.1414\*\* | <0.001 | +0.1134 | Tích cực → lợi nhuận cao |

| S\_t ↔ Intraday Range | \*\*-0.1223\*\* | <0.001 | -0.0756 | Tích cực → biến động thấp |

| \\|S\_t\\| ↔ log(Volume) | \*\*+0.1595\*\* | <0.001 | +0.1607 | Tin mạnh → thanh khoản cao |

| S\_t ↔ \\|Return\\| | -0.0770 | <0.001 | -0.0544 | Tích cực → ít biến động |

| S\_t ↔ Volume | +0.0612 | 0.002 | +0.0577 | Tích cực → thanh khoản tăng |

| \\|S\_t\\| ↔ Intraday Range | +0.0519 | 0.010 | +0.0264 | Tin mạnh → biến động cao |

| \\|S\_t\\| ↔ \\|Return\\| | +0.0514 | 0.010 | +0.0203 | Tin mạnh → biến động return |

| S\_t ↔ log(Volume) | +0.0494 | 0.014 | +0.0577 | Xác nhận cặp volume |



\*\*Tất cả 8/8 cặp đều có ý nghĩa thống kê ở mức p < 0.05.\*\*



\## 4.3. Phát hiện chính



\### Phát hiện 1: Sentiment tích cực → return cao, biến động thấp



\- `S\_t ↔ Return`: r = +0.14 (p<0.001) — sentiment tích cực liên quan đến return dương

\- `S\_t ↔ Intraday Range`: r = -0.12 (p<0.001) — sentiment tích cực liên quan đến biến động thấp



→ \*\*Phù hợp lý thuyết Prospect Theory\*\*: nhà đầu tư lạc quan → giao dịch ổn định → giá tăng nhẹ.



\### Phát hiện 2: Magnitude of sentiment quan trọng hơn direction với thanh khoản



\- `S\_t ↔ Volume`: r = +0.06

\- `\\|S\_t\\| ↔ Volume`: r = \*\*+0.16\*\* (mạnh gấp 2.6 lần)



→ \*\*Phù hợp Attention Theory\*\* (Barber \& Odean 2008): thanh khoản phản ứng với \*\*"tin có quan trọng không"\*\*, không phải \*\*"tin tốt hay xấu"\*\*.



\### Phát hiện 3: Effect size nằm trong literature



\- Sentiment giải thích \~2% variance return daily (r² = 0.02)

\- So sánh: Tetlock (2007) US: |r|=0.05-0.10; Loughran-McDonald (2011): |r|=0.03-0.08

\- \*\*Đồ án VN Banking: |r|=0.14\*\* — vượt literature do banking sector nhạy cảm chính sách



\## 4.4. Hạn chế



1\. \*\*S\_t từ Sentiment\_NLP\*\* (Gemini LLM-labeled), không phải expert-labeled → có thể có noise

2\. \*\*Correlation đồng thời\*\*, không phải causality (cần Granger test bổ sung)

3\. \*\*Banking sector\*\* có thể đặc thù — không generalize cho toàn thị trường

4\. \*\*Daily aggregation\*\* mất intraday dynamics (sentiment sáng vs chiều)



\## 4.5. Hướng mở rộng



1\. \*\*Granger causality test\*\* để kiểm định quan hệ nhân quả

2\. \*\*EGARCH-X\*\* để mô hình hóa volatility có exogenous sentiment

3\. \*\*S\_t từ FSE model\*\* (PhoBERT 3-class, F1=0.XX) khi có đủ corpus lịch sử

4\. \*\*Event study\*\* quanh các sự kiện lớn (Fed, NHNN, earnings)



\## 4.6. Hình ảnh minh họa



\- `data/processed/correlation/scatter\_S\_t\_vs\_logVolume.png`

\- `data/processed/correlation/scatter\_S\_t\_vs\_Return.png`

\- `data/processed/correlation/scatter\_S\_t\_vs\_Range.png`

\- `data/processed/correlation/scatter\_abs\_S\_t\_vs\_Range.png`

\- `data/processed/correlation/rolling\_correlation.png`

