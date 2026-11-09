# Response Curve Kit — ANN mượt ước lượng hàm phản ứng thị trường với tin tức

`fit_response_curve.py` dùng **1 mạng neuron nhân tạo (ANN)** thực hiện hồi quy
S_t → target và trả ra **hàm số mượt f(S_t)** cùng **đạo hàm bậc 1 (f′)** và
**bậc 2 (f″)** tính chính xác bằng autograd — từ đó đọc ra **tính chất "tới hạn"**
của tin tức: tin tác động mạnh nhất ở đâu, thị trường bắt đầu bão hòa từ đâu,
phi tuyến bắt đầu từ đâu, đâu là điểm uốn giữa hai chế độ phản ứng.

## 1. Kiến trúc & phương pháp

| Thành phần | Cụ thể |
|---|---|
| Mạng | MLP 2 lớp ẩn 32–32, **tanh** (hàm kích hoạt C∞-mượt → f, f′, f″ liên tục) |
| Loss | MSE + `lam2 · mean(f″(x)²)` trên grid cố định (**hình phạt năng lượng cong** — giữ cho đường mượt, không rung) |
| Đạo hàm | `torch.autograd` bậc 2 — **chính xác, không lãi số** |
| Ensemble | 6 seeds → f/f′/f″ trung bình ± σ (dải tin cậy); ngưỡng tính **từng seed rồi lấy median** (tránh bị làm phẳng bởi trung bình) |
| Input features | `S_t, delta_S_t, return_lag, vol_lag` (đều z-score) |
| Đường đọc | Partial response: quét S_t trên grid, các feature khác giữ ở median |
| Kiểm định | Chia theo thời gian: 80% đầu train (5% cuối dùng early-stop), **20% cuối làm test không tham gia train**; so R² ANN vs OLS cùng features |
| Ngưỡng | Peak = argmax\|f′\|; sat50/sat20 = \|f′\| giảm về 50%/20% đỉnh; onset = lệch OLS > 1σ ensemble; điểm uốn = f″ đổi dấu; attention onset = f(\|x\|) vượt f(0)+1σ |

**Selftest kèm sẵn** (bắt buộc chạy 1 lần sau khi cài): dữ liệu tổng hợp có đáp án
biết trước `f = 0.9·tanh(2.2x) + 0.25x` (sat50=0.44, sat20=0.84, f′max=2.23 tại 0).
Model phải tái tạo đúng trong dung sai → chứng minh pipeline đọc đúng hình dạng + ngưỡng.

## 2. Cài đặt & chạy

```powershell
# trong venv của project (torch đã có từ khi chạy API; nếu thiếu thì cài)
pip install torch pandas pyarrow numpy scikit-learn matplotlib

# 1) Kiểm chứng pipeline (chạy 1 lần)
python scripts/fit_response_curve.py --selftest

# 2) Chạy thật trên dữ liệu Vietstock (K4 refresh)
python scripts/fit_response_curve.py
#   -> đọc data/processed/banking_with_sentiment_vietstock.parquet (mặc định)
#   -> ra data/processed/response_curves/

# Chọn riêng nguồn S_t / target nếu muốn
python scripts/fit_response_curve.py --sources Gemini --targets avg_return,log_volume

# Không cần biểu đồ (thiếu matplotlib)
python scripts/fit_response_curve.py --skip-plot

# Độ mượt: lam2 lớn = mượt hơn nhưng dễ làm phẳng mất ngưỡng (mặc định 0.0005)
python scripts/fit_response_curve.py --lam2 0.002
```

Nếu file parquet không ở mặc định: `--data đường/dẫn/file.parquet`.

## 3. Đầu ra (data/processed/response_curves/)

| File | Nội dung |
|---|---|
| `curve_<source>_<target>.csv` | Grid 401 điểm: `S_t_raw, S_t_z, f_raw, f_lo/f_hi, fprime_raw(+lo/hi), fpp_raw(+lo/hi), f_lin_raw` |
| `plot_<source>_<target>.png` | 3 panel: f (kèm data + OLS), f′ (kèm cờ ngưỡng), f″ (kèm điểm uốn) |
| `threshold_summary.csv` | Bảng ngưỡng cho cả 6 đường (2 nguồn × 3 target) |

Cột chính của `threshold_summary.csv`:

| Cột | Ý nghĩa |
|---|---|
| `r2_ann_test` / `r2_ols_test` / `delta_r2` | Sức dự báo ngoài mẫu; **ΔR² > 0 = phi tuyến mang thêm thông tin thực** |
| `shape` | Nhãn hình dạng: `S-shape` (đơn điệu, bão hòa 2 đầu) / `V (|S_t|-shape)` (cực tiểu ở trung tính) / `phang` |
| `S_t_peak_raw` | Vị trí **đỉnh nhạy cảm** — tin thay đổi target mạnh nhất ở mức sentiment này |
| `fprime_peak_raw` | Hiệu ứng biên tối đa (đơn vị target / 1 đơn vị S_t) |
| `S_t_sat50_raw`, `S_t_sat20_raw` | Bão hòa 50% / 20%: từ đây thêm tin thì phản ứng tăng thêm ít/ không đáng kể |
| `sat50_spread_z`, `sat20_spread_z` | Độ lệch chuẩn ngưỡng giữa các seed = bất định của ước lượng |
| `S_t_attention_raw` | (cho dạng V) \|S_t\| mà target bắt đầu vượt nền trung tính — **ngưỡng chú ý/thanh khoản** |
| `S_t_onset_raw` | Mức S_t mà hàm bắt đầu lệch rõ khỏi tuyến tính |
| `n_inflections`, `inflection_z` | Số điểm uốn (f″ đổi dấu) và vị trí |
| `f_at_0_raw`, `f_at_plus_m_raw`, `f_at_minus_m_raw`, `asymmetry_raw` | Mức nền trung tính, phản ứng 2 phía, **bất đối xứng tin tốt vs tin xấu** |

## 4. Cách đọc "tính chất tới hạn" — mẫu diễn giải cho báo cáo

**Dạng S-shape (thường gặp với avg_return):**
- f′ đỉnh gần S_t=0 → *tin vừa rời trạng thái trung tính có sức thiểm động giá lớn nhất*.
- f′ suy giảm về 0 hai phía, sat50/sat20 đánh dấu vùng bão hòa → *thị trường hấp thụ tin:
  quá |S_t|* thì tin thêm không kéo giá thêm — đây chính là "tới hạn" của thông tin*.
- f″ đổi dấu tại điểm uốn → *chuyển từ chế độ phản ứng tăng tốc (convex) sang giảm tốc
  (concave); điểm uốn là biên giới hai chế độ*.
- `asymmetry_raw` ≠ 0 → *phản ứng với tin cực tính (+) và (−) không đối xứng*.

**Dạng V (thường gặp với log_volume / avg_range):**
- Cực tiểu tại trung tính, tăng theo |S_t| cả 2 phía → *không phải chiều mà là CƯỜNG ĐỘ
  tin quyết định; đúng trực giác "tin cực đoan dù tốt hay xấu đều làm sôi giao dịch"*.
- Đọc `S_t_attention_raw` thay cho sat: ngưỡng |S_t| mà khối lượng/biến động rời nền →
  *ngưỡng chú ý của thị trường*.
- ΔR² lớn so với OLS (không thể khớp V bằng đường thẳng) → *bằng chứng phi tuyến mạnh*.

**Ba câu "trả về ý nghĩa thực sự" cho phần kết luận:**
1. *Tác động biên* của tin tức không phải hằng số — nó là hàm của cường độ tin (f′ phụ thuộc vị trí).
2. Tồn tại *vùng tới hạn*: dưới |S_t|* thị trường vẫn còn "chưa hợp thức hóa" tin; trên
   sat-threshold thị trường đã hấp thụ xong — thông tin dư không đổi giá/khối lượng thêm đáng kể.
3. Phi tuyến này có giá trị dự báo ngoài mẫu (ΔR² > 0 trên 20% cuối kỳ) — không chỉ là
   hiện tượng khớp đường.

## 5. Ghi chú học thuật (cho phần phương pháp của báo cáo)

- Ý tưởng cùng họ với **threshold regression (Hansen 2000)** và **STAR** (mô hình chuyển
  chế độ mượt) nhưng ước lượng bằng ANN mượt + hình phạt f″ (bending energy, tương tự
  natural spline) — không cần chọn trước số ngưỡng hay hàm chuyển tiếp.
- Đây là phân tích **tương quan có cấu trúc, chưa phải nhân quả**: f(S_t) là hàm phản ứng
  điều kiện, có kiểm soát return_lag/vol_lag nhưng chưa kiểm soát hết biến nhiễu.
- Ngưỡng có bất định: luôn báo kèm `*_spread_z` và dải ±1σ trong biểu đồ.
- Cửa sổ dữ liệu: 2022-01 → 2026-11 (giá Vietstock 26 mã ngân hàng); S_t từ 2 nguồn
  độc lập (Gemini từ DB VN-Quant, Model từ PhoBERT best.pt).

## 6. Nếu số seed/độ mượt cần chỉnh

Sửa hằng số đầu file: `N_SEEDS`, `MAX_EPOCHS`, `PATIENCE`, `HIDDEN`, `LR`,
`WEIGHT_DECAY`, `PEN_LO/PEN_HI`. Mặc định đã được hiệu chỉnh qua selftest: λ2 quá lớn
(≥0.01) sẽ làm phẳng mất ngưỡng; quá nhỏ (≈0) cho đường hơi rung ở mép dữ liệu.
