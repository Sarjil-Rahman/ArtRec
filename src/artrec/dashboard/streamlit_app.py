from __future__ import annotations
from pathlib import Path
import json
import streamlit as st
from artrec.data.io import read_csv_with_types
from artrec.serve.recommender import RecommendationService

st.set_page_config(page_title="ArtRec Demo", layout="wide")
st.title("ArtRec Production Demo")
data_dir = Path("data")
artifact_dir = Path("artifacts")
if not (data_dir / "raw" / "catalog.csv").exists():
    st.error("Build the pipeline first with: python scripts/build_all.py")
    st.stop()
catalog_df = read_csv_with_types(data_dir / "raw" / "catalog.csv")
users_df = read_csv_with_types(data_dir / "raw" / "users.csv")
service = RecommendationService.load(
    catalog_path=data_dir / "raw" / "catalog.csv",
    users_path=data_dir / "raw" / "users.csv",
    retriever_path=artifact_dir / "retriever.joblib",
    ranker_path=artifact_dir / "ranker.joblib",
    impressions_path=data_dir / "raw" / "impressions.csv",
)
metric_path = artifact_dir / "business_metrics.json"
if metric_path.exists():
    metrics = json.loads(metric_path.read_text(encoding="utf-8"))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CTR", f"{metrics['ctr']:.2%}")
    c2.metric("Cart rate", f"{metrics['cart_rate']:.2%}")
    c3.metric("Revenue", f"GBP {metrics['revenue']:,.0f}")
    c4.metric("Margin", f"GBP {metrics['margin']:,.0f}")
st.subheader("Personalized recommendations")
user_id = st.selectbox("User", users_df["user_id"].sort_values().tolist()[:100])
limit = st.slider("How many recommendations?", min_value=5, max_value=20, value=12)
if st.button("Generate recommendations"):
    st.dataframe(service.recommend(user_id=user_id, limit=limit))
st.subheader("Catalog sample")
st.dataframe(catalog_df.head(30))
