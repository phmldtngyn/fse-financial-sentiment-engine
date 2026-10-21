"""Streamlit dashboard — FSE Engine.

Chạy: streamlit run dashboard/app.py
Mở: http://localhost:8501
"""
import streamlit as st
import httpx
import os
API = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="FSE Engine",
    page_icon="📊",
    layout="wide",
)

st.title("📊 FSE Engine — Financial Sentiment & Liquidity Forecast")
st.caption("Phân tích cảm xúc tin tức tài chính & dự báo thanh khoản thị trường VN")


# Health check
try:
    r = httpx.get(f"{API}/health", timeout=3)
    health = r.json()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sentiment", "✅" if health["models_loaded"]["sentiment"] else "❌")
    col2.metric("RAG", "✅" if health["models_loaded"]["rag"] else "❌")
    col3.metric("Forecast", "✅" if health["models_loaded"]["forecast"] else "❌")
    col4.metric("Metrics", "✅" if health["models_loaded"]["metrics"] else "❌")
except Exception as e:
    st.error(f"Không kết nối được API tại {API}. Đã chạy `uvicorn api.main:app` chưa?")
    st.code(str(e))
    st.stop()

st.divider()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["💭 Sentiment", "🔍 RAG", "📈 Forecast", "📊 Metrics"])

# Tab 1: Sentiment
with tab1:
    st.header("Chấm điểm cảm xúc")
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
            col1.metric("Score", f"{res['score']:+.3f}", help="∈ [-1, +1]")
            col2.metric("Tier", res["tier"], help="1=không liên quan, 3=trực tiếp")
            if res["score"] > 0.3:
                st.success("Tin tích cực")
            elif res["score"] < -0.3:
                st.error("Tin tiêu cực")
            else:
                st.info("Tin trung tính")
        except Exception as e:
            st.error(f"Lỗi: {e}")

# Tab 2: RAG
with tab2:
    st.header("Truy vấn ngữ cảnh RAG")
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
                st.warning("Không tìm thấy kết quả. Index có thể trống hoặc window quá hẹp.")
            for i, res in enumerate(results, 1):
                with st.expander(f"[{i}] score={res['score']:.3f} (sim={res['sim']:.3f}, {res['age_hours']:.0f}h trước)"):
                    st.write(res["text"])
        except Exception as e:
            st.error(f"Lỗi: {e}")

# Tab 3: Forecast
with tab3:
    st.header("Dự báo thanh khoản")
    col1, col2 = st.columns(2)
    with col1:
        S_t = st.number_input("S_t (sentiment)", value=0.5, min_value=-1.0, max_value=1.0, step=0.1)
        vol_lag = st.number_input("Vol_{t-1}", value=1_200_000, step=100_000)
    with col2:
        delta_S_t = st.number_input("ΔS_t", value=0.0, min_value=-1.0, max_value=1.0, step=0.1)
        return_lag = st.number_input("Return_{t-1}", value=0.0, min_value=-0.1, max_value=0.1, step=0.01)

    if st.button("Dự báo", key="btn_forecast"):
        try:
            r = httpx.post(
                f"{API}/api/v1/forecast",
                json={
                    "S_t": S_t, "delta_S_t": delta_S_t,
                    "vol_lag": vol_lag, "return_lag": return_lag,
                    "v_ctx": [],
                },
                timeout=30,
            )
            r.raise_for_status()
            res = r.json()
            st.metric("Dự báo Vol_t", f"{res['y_pred_vol']:,.0f}")
            st.caption(f"Model: {res['model']}")
        except Exception as e:
            st.error(f"Lỗi: {e}")

# Tab 4: Metrics
with tab4:
    st.header("So sánh 4 model × 3 scenario")
    try:
        r = httpx.get(f"{API}/api/v1/metrics", timeout=10)
        r.raise_for_status()
        metrics = r.json()

        import pandas as pd
        rows = []
        for scenario, models in metrics.items():
            for model, m in models.items():
                rows.append({
                    "Scenario": scenario,
                    "Model": model,
                    "RMSE": m["rmse"],
                    "MAE": m["mae"],
                    "R²": m["r2"],
                })
        df = pd.DataFrame(rows)
        st.dataframe(
            df.pivot_table(index="Model", columns="Scenario", values="RMSE").round(4),
            use_container_width=True,
        )

        st.subheader("Chi tiết")
        st.dataframe(df.round(4), use_container_width=True)
    except Exception as e:
        st.error(f"Lỗi: {e}")