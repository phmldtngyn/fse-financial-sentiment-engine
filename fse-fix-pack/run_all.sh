#!/usr/bin/env bash
# run_all.sh — chạy tuần tự 9 script của FSE Depth Extension Pack.
#
# Cách dùng:
#   cd /path/to/fse-financial-sentiment-engine
#   bash run_all.sh
#
# Mỗi script tự exit code != 0 nếu có lỗi nghiêm trọng. Để tiếp tục dù lỗi,
# set SKIP_ON_ERROR=1:
#   SKIP_ON_ERROR=1 bash run_all.sh
#
# Mỗi script ghi output vào data/processed/<module>/, đọc README.md để xem chi tiết.

set -u  # unset variable = error
SKIP_ON_ERROR="${SKIP_ON_ERROR:-0}"

repo_root="$(cd "$(dirname "$0")" && pwd)"
cd "$repo_root"

echo "============================================================"
echo "FSE Engine — Depth Extension Pack"
echo "Repo root: $repo_root"
echo "Started  : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

run_step() {
    local num="$1"; shift
    local name="$1"; shift
    echo ""
    echo "=== [$num] $name ==="
    python "scripts/${num}_${name}.py" "$@"
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "[!] Script $num exited with code $rc"
        if [ "$SKIP_ON_ERROR" = "1" ]; then
            echo "    SKIP_ON_ERROR=1 → tiếp tục."
        else
            echo "    Dừng. Để tiếp tục: SKIP_ON_ERROR=1 bash run_all.sh"
            exit $rc
        fi
    fi
    return 0
}

# Tier 1 — bắt buộc cho credibility
run_step 01 evaluate_model
run_step 02 compare_sentiments
run_step 03 robustness_battery

# Tier 2 — nhân quả
run_step 04 granger_causality
run_step 05 var_irf

# Tier 3 — volatility
run_step 06 egarch_x

# Tier 5 — incremental informativeness (QUAN TRỌNG NHẤT)
run_step 07 partial_correlation

# Tier 4 — event study
run_step 08 event_study

# Tier 6 — forecasting
run_step 09 forecast_comparison

echo ""
echo "============================================================"
echo "Hoàn tất: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Đọc các *_summary.json trong data/processed/*/ để có số liệu đưa vào báo cáo."
echo "============================================================"
