# Phương pháp luận mở rộng — Depth Extensions

Tài liệu này giải thích lý do phương pháp, citation, và cách đưa kết quả vào bài
nghiên cứu. Mỗi section tương ứng 1 script.

---

## 01 — Đánh giá mô hình (F1, confusion matrix, Cohen's kappa)

### Vấn đề
README hiện ghi `F1-macro = 0.XX` (placeholder). Bài nghiên cứu không thể công bố
khi chỉ số chất lượng mô hình chưa có số thực.

### Phương pháp
- **F1-macro**, **F1-weighted**, **F1 per-class** trên test set độc lập.
- **Balanced accuracy**: xử lý imbalance.
- **Cohen's Kappa** (κ): so với random guess, xử lý agreement do may rủi.
- **Confusion matrix**: xem lớp nào bị confuse nhiều (thường là neutral).
- **Bootstrap 95% CI** cho F1-macro (500 resample).
- **Random baseline**: F1-macro khi dự đoán uniform random ≈ mean(2p_i/(1+p_i))
  với p_i = prevalence của class i. So sánh improvement = F1-macro - F1_random.

### Citation
- Cohen, J. (1960). *A coefficient of agreement for nominal scales.* Educational and Psychological Measurement, 20(1), 37–46.
- Powers, D. M. W. (2011). *Evaluation: from precision, recall and F-measure to ROC, informedness, markedness and correlation.* J. Machine Learning Tech. 2(1), 37–63.

### Cách đưa vào báo cáo
Bảng trong phần "4. Đánh giá mô hình":

| Chỉ số | Giá trị | 95% CI | Baseline |
|---|---|---|---|
| F1-macro | 0.72 | [0.69, 0.74] | 0.41 |
| Accuracy | 0.78 | — | 0.33 |
| Cohen's κ | 0.61 | — | 0.00 |

---

## 02 — So sánh hai nguồn sentiment (PhoBERT vs Gemini)

### Vấn đề
Phân tích tương quan đang dùng `Sentiment_NLP` của VN-Quant (nhãn LLM Gemini),
không phải S_t do mô hình PhoBERT của bài tạo ra. Phản biện sẽ hỏi: "vậy PhoBERT
để làm gì?".

### Phương pháp
- Sinh `S_t_model` = mean(prob_pos - prob_neg) per day từ output PhoBERT.
- Tính Pearson + Spearman giữa S_t_gemini và S_t_model.
- Chạy lại 8 cặp tương quan với cả 2 nguồn.
- Báo cáo: số cặp cùng dấu, số cặp cùng significant, mean|r_diff|.

### Tiêu chí đánh giá
- |r| > 0.7 và ≥ 80% cặp cùng significant → hai nguồn tương đương, dùng PhoBERT
  (tái lập, rẻ, không phụ thuộc API bên thứ 3).
- |r| < 0.3 → hai nguồn đo thứ khác nhau → cần điều tra (có thể PhoBERT kém Gemini
  trên tài chính tiếng Việt).

### Citation
- Tetlock, P. C. (2007). *Giving content to investor sentiment.* J. Finance, 62(3), 1139–1168.
- Loughran, T., & McDonald, B. (2011). *When is a liability not a liability?* J. Finance, 66(1), 35–65.

---

## 03 — Robustness battery

### Vấn đề
Bảng 8 cặp tương quan chỉ có 1 con số r và 1 p-value. Phản biện sẽ hỏi:
- CI của r là gì?
- Có ổn định qua giai đoạn không?
- Có điểm gãy cấu trúc không?

### Ba phép thử
1. **Block bootstrap CI** (Politis & White 2004): block size L=5 (1 tuần giao dịch),
   n_boot=5000. Lý do: chuỗi tài chính có autocorrelation, bootstrap i.i.d. làm
   mất cấu trúc → CI thiên lệch.
2. **Subsample stability**: chia 2016–2019 / 2020–2022 / 2023–2026 (3 giai đoạn:
   pre-COVID bình thường, COVID+post-COVID, giai đoạn gần đây). Cặp ổn định = cùng
   dấu significant trong cả 3 giai đoạn.
3. **Bai–Perron (1998, 2003) multiple breakpoints**: tối đa 3 điểm gãy, min_seg=60,
   supF test với ngưỡng Bai (1997, Table 1) cho p=2.
4. **CUSUM of squares (Brown–Durbin–Evans 1975)**: phát hiện break tổng quát.
5. **Rolling correlation 120 ngày**: visualize stability.

### Tiêu chí
Bảng đủ robust nếu:
- ≥ 80% cặp có CI excludes zero
- ≥ 80% cặp stable sign across 3 subsamples
- ≤ 30% cặp có structural break

### Citation
- Politis, D. N., & White, H. (2004). *Automatic block-length selection for the dependent bootstrap.* Econometric Reviews, 23(1), 53–70.
- Bai, J., & Perron, P. (2003). *Computation and analysis of multiple structural change models.* J. Applied Econometrics, 18(1), 1–22.
- Brown, R. L., Durbin, J., & Evans, J. M. (1975). *Techniques for testing the constancy of regression relationships over time.* JRSS B, 37(2), 149–192.

---

## 04 — Granger causality + Toda–Yamamoto

### Vấn đề
Tương quan ≠ nhân quả. Phản biện sẽ hỏi: "sentiment thực sự tiên đoán hay chỉ phản
ứng?".

### Hai phép thử
1. **Granger (1969)**: Wald test cho H0: các hệ số lag của x trong phương trình y = 0.
   Yêu cầu chuỗi dừng. Chuỗi tài chính thường có unit root → cần (2).
2. **Toda–Yamamoto (1995)**: ước lượng VAR(p+d) với d = bậc tích hợp tối đa, Wald
   trên p lags đầu. Robust với unit root/cointegration.

### Bidirectional
Chạy cả 2 hướng:
- S_t → target (sentiment tiên đoán giá/khối lượng?)
- target → S_t (giá/khối lượng tiên đoán sentiment — feedback loop?)

### Diễn giải
- Chỉ S → target significant: sentiment có vai trò tiên đoán.
- Cả 2 significant: feedback loop (sentiment vừa phản ứng vừa tiên đoán).
- Chỉ target → S: sentiment chỉ phản ứng, không tiên đoán.

### Citation
- Granger, C. W. J. (1969). *Investigating causal relations by econometric models and cross-spectral methods.* Econometrica, 37(3), 424–438.
- Toda, H. Y., & Yamamoto, T. (1995). *Statistical inference in vector autoregressions with possibly integrated processes.* J. Econometrics, 66(1-2), 225–250.

---

## 05 — Local Projections + VAR IRF

### Vấn đề
Bảng tương quan cho 1 con số r, không cho biết **độ lớn** và **độ bền** của tác động.
IRF cho thấy "1 cú sốc sentiment +1σ → target phản ứng bao nhiêu tại h=1, 3, 7, 14".

### Hai cách ước lượng IRF
1. **Local Projections (Jordà 2005)**: mỗi h là 1 OLS riêng y_{t+h} ~ x_t + controls.
   Ưu điểm: không cần chỉ định đúng VAR, ổn định với phi tuyến.
2. **VAR-Cholesky IRF**: VAR(p) với ordering (S_t, target). Đối chiếu để kiểm tra
   kết quả LP có bị outlier method-specific không.

### Đọc IRF
- IRF[h=0] = phản ứng tức thời.
- Peak IRF tại h>0 = hiệu ứng trễ (ví dụ: sentiment ảnh hưởng return mạnh nhất sau 2 ngày).
- Cumulative IRF tại h=14: tổng tác động 2 tuần. Nếu ≠ 0 → tác động bền.

### Citation
- Jordà, Ò. (2005). *Estimation and inference of impulse responses by local projections.* American Economic Review, 95(1), 161–182.
- Kilian, L., & Lütkepohl, H. (2017). *Structural Vector Autoregressive Analysis.* Cambridge Univ. Press.

---

## 06 — EGARCH-X + GJR-GARCH-X

### Vấn đề
Bảng tương quan có cặp |S_t| ↔ intraday range (+0.05) — cho thấy sentiment có thể
liên quan đến biến động. Nhưng đó là tương quan đồng thời, không vào variance equation.
EGARCH-X cho phép formal kiểm định: sentiment có giải thích variance không?

### Hai mô hình
1. **EGARCH(1,1)-X (Nelson 1991)**:
   ln(σ²_t) = ω + α·|z_{t-1}| + γ·z_{t-1} + β·ln(σ²_{t-1}) + δ·S_{t-1}
   - γ: leverage (tin xấu tăng vol hơn tin tốt cùng độ lớn)
   - δ: sentiment trong variance equation (báo cáo ở mean do arch package support)

2. **GJR-GARCH(1,1)-X (Glosten-Jagannathan-Runkle 1993)**:
   σ²_t = ω + (α + γ·I{ε<0})·ε²_{t-1} + β·σ²_{t-1} + δ·S_{t-1}

### Citation
- Nelson, D. B. (1991). *Conditional heteroskedasticity in asset returns: A new approach.* Econometrica, 59(2), 347–370.
- Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). *On the relation between the expected value and the volatility of the nominal excess return on stocks.* J. Finance, 48(5), 1779–1801.

---

## 07 — Partial correlation + Incremental R² (PHÉP THỬ QUYẾT ĐỊNH)

### Vấn đề
Câu hỏi phản biện số 1: "Sentiment chỉ đại diện cho thị trường chung. Sau khi
kiểm soát VNINDEX và foreign net buying, sentiment còn gì không?".

### Hai phép thử
1. **Partial correlation**: residual(S_t ~ macro) và residual(target ~ macro),
   tính Pearson giữa 2 residual. Nếu còn significant → sentiment có thông tin GIA TĂNG.
2. **Incremental R² F-test**: so sánh R² của:
   - Reduced: target ~ macro
   - Full: target ~ sentiment + macro
   F = ((RSS_reduced - RSS_full) / k_s) / (RSS_full / dof). Nếu p < 0.05 → sentiment
   tăng R² đáng kể.

### Citation
- Tetlock, P. C., Saar-Tsechansky, M., & Macskassy, S. (2008). *More than words: Quantifying language to measure firms' fundamentals.* J. Finance, 63(3), 1437–1467.

### Cách đưa vào báo cáo
Đây là phần **quan trọng nhất** của bài mở rộng — đặt ngay sau bảng 8 cặp tương
quan chính, có tiêu đề "4.4. Sức giải thích gia tăng của sentiment khi kiểm soát
vĩ mô".

---

## 08 — Event study

### Vấn đề
IRF cho tác động trung bình, nhưng phản biện muốn thấy tác động quanh các event
cụ thể (Fed pivot, NHNN rate, earnings, COVID, SVB).

### Phương pháp
1. **Market model** để ước lượng expected return: r_expected = α + β·r_VNINDEX,
   ước từ cửa sổ [-250, -30] ngày trước event.
2. **AR_t** = r_ticker,t - r_expected,t
3. **CAR[1, +post]** = Σ AR_t trong [+1, +window_post]
4. **t-test** cho AAR từng ngày (average across tickers).
5. **Cross-sectional regression**: CAR[1, post] ~ S_t[-pre, 0]. Nếu sentiment
   trong [-pre, 0] tiên đoán CAR[1, post] → sentiment có giá trị thực tiễn.
6. **Direction match**: S_t pre-event có khớp chiều kỳ vọng với loại event
   (positive event → S_t > 0) không?

### Citation
- MacKinlay, A. C. (1997). *Event studies in economics and finance.* J. Economic Literature, 35(1), 13–39.
- Boehmer, E., Musumeci, J., & Poulsen, A. B. (1991). *Event-study methodology under conditions of event-induced variance.* J. Financial Economics, 30(2), 253–272.

---

## 09 — Forecast comparison + Diebold–Mariano / Clark–West

### Vấn đề
Tương quan significant ≠ có giá trị dự báo out-of-sample. Phải kiểm chứng bằng
walk-forward forecasting.

### Ba mô hình
1. **Baseline**: AR(5) + day-of-week dummies.
2. **+ S_t (Gemini)**: AR(5) + DoW + S_t từ VN-Quant.
3. **+ S_t (model)**: AR(5) + DoW + S_t từ PhoBERT.

### Hai phép thử
1. **Diebold–Mariano (1995)**: so sánh dự báo accuracy giữa 2 mô hình (non-nested).
   DM < 0 và p < 0.05 → baseline sai hơn (sentiment có giá trị).
2. **Clark–West (2007)**: test đúng cho nested models (baseline ⊂ +S_t). CW
   hiệu chỉnh bias do baseline dùng chung một phần variance với full model.

### Hit rate
Tỷ lệ dự báo đúng chiều (sign) của return. > 55% → có giá trị kinh tế (long/short
signal).

### Citation
- Diebold, F. X., & Mariano, R. S. (1995). *Comparing predictive accuracy.* J. Business & Economic Statistics, 13(3), 253–263.
- Clark, T. E., & West, K. D. (2007). *Approximately normal tests for equal predictive accuracy in nested models.* J. Econometrics, 138(1), 291–311.

---

## Thứ tự ưu tiên khi đưa vào bài báo

| Section trong báo cáo | Script cung cấp số liệu |
|---|---|
| 4.1 Đánh giá mô hình | 01 |
| 4.2 Bảng tương quan 8 cặp | (đã có từ correlation_analysis.py) + 03 cho CI/breaks |
| 4.3 So sánh sentiment PhoBERT vs Gemini | 02 |
| 4.4 Sức giải thích gia tăng khi kiểm soát vĩ mô | 07 |
| 4.5 Phân tích nhân quả | 04 + 05 (IRF plot) |
| 4.6 EGARCH-X: sentiment trong variance | 06 |
| 4.7 Event study: sentiment quanh các sự kiện | 08 |
| 4.8 Out-of-sample: sentiment có giá trị dự báo không? | 09 |
| Appendix A: Robustness battery | 03 (bootstrap CI, subsample, breaks) |
| Appendix B: Phương pháp chi tiết | docs/depth_extensions.md |

Phần **4.4** (script 07) nên đặt ngay sau 4.2 để trả lời phản biện sớm nhất.
