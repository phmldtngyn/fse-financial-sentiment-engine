# FSE Colab Kit — Tích hợp model tự train

Bộ file đi kèm tài liệu `FSE_HuongDan_Colab_TichHop_KiemDinh.docx`.

| File | Dùng ở | Làm gì |
|---|---|---|
| `FSE_PhoBERT_Colab.ipynb` | Phase B+C (Colab) | Upload lên Google Colab → chạy theo thứ tự cell |
| `export_texts_with_dates.py` | Phase A (máy local) | Chạy từ gốc repo → xuất `texts_with_dates.csv` → upload Drive |
| `api_routes_sentiment.py` | Phase D (máy local) | Thay thế `api/routes/sentiment.py` — nạp checkpoint mới qua `load_checkpoint` |
| `test_trained_model.py` | Phase D (máy local) | Thay thế `scripts/test_trained_model.py` — test 5 câu trên model tự train |
| `gitignore_FIXED.txt` | Phase D (máy local) | Thay thế `.gitignore` (bản hiện tại lỡ chứa script PowerShell nên rule không chạy) |

Sau khi thay 2 file patch: `git mv`/đổi tên đúng đường dẫn trong repo rồi commit.
Chi tiết từng lệnh nằm ở Phase D trong tài liệu docx.
