"""Streamlit dashboard — FSE Engine.

Chạy: streamlit run dashboard/app.py
Mở: http://localhost:8501
"""
import os
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="FSE Engine",
    page_icon="📊",
    layout="wide",
)

st.title("📊 FSE Engine — Financial Sentiment & Banking Correlation")
st.caption(
    "Phân tích cảm xúc tin tức tài chính & tương quan với biến động "
    "nhóm cổ phiếu ngân hàng Việt Nam"
)

# === Health check ===
try:
    r = httpx.get(f"{API}/health", timeout=3)
    health = r.json()
    c1, c2, c3 = st.columns(3)
    c1.metric("Sentiment", "✅" if health["models_loaded"]["sentiment"] else "❌")
    c2.metric("RAG", "✅" if health["models_loaded"]["rag"] else "❌")
    c3.metric("Correlation", "✅" if health["models_loaded"]["correlation"] else "❌")
except Exception as e:
    st.error(f"Không kết nối được API tại {API}. Đã chạy `python -m uvicorn api.main:app` chưa?")
    st.code(str(e))
    st.stop()

st.divider()

# === Tabs ===
tab1, tab2, tab3 = st.tabs(["💭 Sentiment", "🔍 RAG", "🏦 Banking Correlation"])


# ============================================================
# Tab 1: Sentiment
# ============================================================
with tab1:
    st.header("Chấm điểm cảm xúc tin tức")
    st.caption("Model: PhoBERT-base-v2 3-class (negative / neutral / positive)")

    text = st.text_area(
        "Nhập bài báo / câu cần phân tích",
        value="Cổ phiếu VNM tăng trần, khối ngoại mua ròng 200 tỷ đồng",
        height=120,
    )

    if st.button("Phân tích", key="btn_sentiment"):
        try:
            r = httpx.post(f"{API}/api/v1/sentiment", json={"text": text}, timeout=30)
            r.raise_for_status()
            res = r.json()

            col1, col2 = st.columns(2)
            col1.metric("Score", f"{res['score']:+.2f}", help="negative=-0.7, neutral=0.0, positive=+0.7")
            col2.metric("Tier", res["tier"], help="1=không liên quan, 3=trực tiếp")

            if res["score"] > 0.3:
                st.success("📈 Tin tích cực")
            elif res["score"] < -0.3:
                st.error("📉 Tin tiêu cực")
            else:
                st.info("➡️ Tin trung tính")
        except Exception as e:
            st.error(f"Lỗi: {e}")


# ============================================================
# Tab 2: RAG
# ============================================================
with tab2:
    st.header("Truy vấn ngữ cảnh RAG")
    st.caption("FAISS rolling index — top-K contexts có time-weighted rerank")

    query = st.text_input("Query", value="cổ phiếu ngân hàng tăng trần")
    top_m = st.slider("Số kết quả", 1, 10, 5)

    if st.button("Tìm", key="btn_rag"):
        try:
            r = httpx.post(
                f"{API}/api/v1/rag/query",
                json={"query": query, "top_m": top_m, "window_hours": 720},
                timeout=30,
            )
            r.raise_for_status()
            results = r.json()

            if not results:
                st.warning("Không tìm thấy kết quả. Index có thể trống.")
            else:
                for i, res in enumerate(results, 1):
                    with st.expander(
                        f"[{i}] score={res['score']:.3f} "
                        f"(sim={res['sim']:.3f}, {res['age_hours']:.0f}h trước)"
                    ):
                        st.write(res["text"])
        except Exception as e:
            st.error(f"Lỗi: {e}")


# ============================================================
# Tab 3: Banking Correlation
# ============================================================
with tab3:
    st.header("Tương quan: Sentiment vs Biến động Ngân hàng")
    st.caption(
        "S_t (Sentiment_NLP) vs thanh khoản/return/biến động 26 mã ngân hàng — "
        "2,492 ngày (2016-2026)"
    )

    try:
        r = httpx.get(f"{API}/api/v1/correlation", timeout=10)
        r.raise_for_status()
        results = r.json()

        df = pd.DataFrame(results)

        # === Bảng correlation ===
        st.subheader("📋 Bảng tương quan")
        df_display = df[["pair", "pearson_r", "pearson_p", "spearman_r", "spearman_p", "n"]].copy()
        df_display.columns = ["Cặp", "Pearson r", "p-value", "Spearman ρ", "p-value ρ", "n"]
        df_display["Sig"] = df["significant"].map({True: "✅ ***", False: "❌"})
        st.dataframe(df_display, use_container_width=True)

        # === Diễn giải ===
        st.subheader("📖 Diễn giải chính")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **① Sentiment tích cực → return cao, biến động thấp**
            - `S_t ↔ Return`: r = **+0.14** (p<0.001)
            - `S_t ↔ Intraday Range`: r = **-0.12** (p<0.001)
            
            → Phù hợp Prospect Theory: nhà đầu tư lạc quan → giá tăng ổn định
            """)
        with col2:
            st.markdown("""
            **② Magnitude quan trọng hơn direction với thanh khoản**
            - `S_t ↔ Volume`: r = +0.06
            - `|S_t| ↔ Volume`: r = **+0.16** (mạnh 2.6x)
            
            → Phù hợp Attention Theory: thanh khoản phản ứng với "tin quan trọng", không phải "tin tốt/xấu"
            """)

        # === Biểu đồ ===
        st.subheader("📈 Biểu đồ scatter")
        corr_dir = Path("data/processed/correlation")
        if corr_dir.exists():
            scatter_files = sorted(corr_dir.glob("scatter_*.png"))
            if scatter_files:
                cols = st.columns(2)
                for i, png in enumerate(scatter_files):
                    with cols[i % 2]:
                        title = png.stem.replace("scatter_", "").replace("_", " ")
                        st.image(str(png), caption=title, use_container_width=True)

            rolling_png = corr_dir / "rolling_correlation.png"
            if rolling_png.exists():
                st.subheader("📉 Rolling 60-day Correlation")
                st.image(str(rolling_png), use_container_width=True)
        else:
            st.info("Chưa có biểu đồ. Chạy: `python -m scripts.correlation_analysis`")

        # === Kết luận ===
        st.subheader("🎯 Kết luận")
        n_sig = sum(1 for x in results if x["significant"])
        st.success(f"""
        **8/8 cặp tương quan đều có ý nghĩa thống kê (p<0.05)** trên 2,492 ngày giao dịch.
        
        - Sentiment giải thích ~**2% variance** return daily (r² = 0.02)
        - Effect size nằm trong literature: Tetlock (2007) |r|=0.05-0.10, **đồ án đạt |r|=0.14**
        - Magnitude of sentiment `|S_t|` là **signal mạnh nhất** với thanh khoản (r=0.16)
        """)

    except Exception as e:
        st.error(f"Lỗi: {e}")
        st.code("Chạy: python -m scripts.correlation_analysis")


# === Footer ===
st.divider()
st.caption(
    "FSE Engine v2.0 · "
    "PhoBERT-base-v2 3-class · "
    "RAG FAISS rolling · "
    "Banking correlation analysis"
)