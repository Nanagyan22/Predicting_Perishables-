import streamlit as st
import pandas as pd
import pickle
import json
import numpy as np
import os
import joblib
import xgboost as xgb
from typing import Dict, Any, List, Tuple
from math import ceil
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv
from importlib import util as import_util
from gemini import chat_with_frostmart, load_knowledge_base


# home.py
from gemini import chat_with_frostmart, generate_frostmart_report

# Load knowledge base
with open("inference/frostmart_knowledge_base.md", "r", encoding="utf-8") as f:
    frostmart_knowledge_base = f.read()



load_dotenv()

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="FrostMart Perishable Demand Prediction System",
    page_icon="🥕",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------
# Custom Header
# -------------------------
st.markdown("""
<div class="main-header">
    <h1>🥕 FROSTMART</h1>
    <h3>Perishable Demand Prediction System</h3>
    <p>Demand forecasting & waste reduction</p>
    <p style="font-size: 14px; margin-top: 10px;">Applied Machine Learning for Inventory & Procurement</p>
</div>
<style>
.main-header {
    text-align: center;
    padding: 20px 0;
    background: linear-gradient(90deg, #0b6b3a, #2aa876);
    color: white;
    margin: -30px -30px 30px -30px;
    border-radius: 0 0 10px 10px;
}
.stAlert > div { padding: 1rem; }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Load Model + Assets
# -------------------------
INFERENCE_DIR = "inference"

@st.cache_resource
def load_assets(inference_dir: str = INFERENCE_DIR) -> Tuple[Any, List[str], Any]:
    model_path = os.path.join(inference_dir, "xgboost_model.pkl")
    features_path = os.path.join(inference_dir, "xgboost_features.json")
    enc_path = os.path.join(inference_dir, "encoding_artifacts.pkl")

    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        st.stop()
    model = None
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except Exception:
        try:
            model = joblib.load(model_path)
        except Exception:
            booster = xgb.Booster()
            booster.load_model(model_path)
            model = booster

    if not os.path.exists(features_path):
        st.error(f"❌ Missing file: {features_path}")
        st.stop()
    with open(features_path, "r") as f:
        feature_names = json.load(f)
    if not isinstance(feature_names, list):
        st.error("❌ xgboost_features.json must contain a list.")
        st.stop()

    encodings = None
    if os.path.exists(enc_path):
        try:
            with open(enc_path, "rb") as f:
                encodings = pickle.load(f)
        except Exception:
            try:
                encodings = joblib.load(enc_path)
            except Exception:
                encodings = None

    return model, feature_names, encodings


model, feature_names, encodings = load_assets()

# -------------------------
# Helper Functions
# -------------------------
def prepare_input_data(input_dict: Dict[str, Any]) -> pd.DataFrame:
    prepared = {feat: 0.0 for feat in feature_names}
    for k, v in input_dict.items():
        if k in prepared:
            prepared[k] = v
    return pd.DataFrame([prepared], columns=feature_names)

def encode_input_data(df: pd.DataFrame, encodings: Any = None) -> pd.DataFrame:
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include=["object"]).columns:
        if encodings and isinstance(encodings, dict) and col in encodings:
            encoder = encodings[col]
            try:
                df_encoded[col] = encoder.transform(df_encoded[col].astype(str))
            except Exception:
                known = getattr(encoder, "classes_", None)
                if known is not None:
                    df_encoded[col] = df_encoded[col].apply(lambda x: int(np.where(known == str(x))[0][0]) if str(x) in known else 0)
                else:
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        else:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    return df_encoded

def make_prediction(input_df: pd.DataFrame) -> float:
    df_encoded = encode_input_data(input_df, encodings)
    try:
        if isinstance(model, xgb.Booster):
            dmatrix = xgb.DMatrix(df_encoded.values, feature_names=list(df_encoded.columns))
            preds = model.predict(dmatrix)
        else:
            preds = model.predict(df_encoded)
        return float(np.array(preds)[0])
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        return None

def suggested_order(predicted_units: float, buffer_pct: float = 0.05) -> int:
    try:
        return int(ceil(predicted_units * (1 + buffer_pct)))
    except Exception:
        return 0

# -------------------------
# Layout Columns
# -------------------------
left_col, chat_col = st.columns([3, 1])

# ===============================================================
# RIGHT SIDE: FROSTMART UK AI CHAT ASSISTANT
# ===============================================================
with chat_col:
    st.markdown("### 🤖 FrostMart AI Chat Assistant")
    st.markdown("*Ask data-driven questions about sales, wastage, pricing, or forecasting performance.*")

    # Initialize session
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Chat container
    chat_container = st.container()
    with chat_container:
        st.markdown("<div style='height:700px; overflow-y:auto; border:1px solid #ddd; padding:10px; border-radius:10px;'>", unsafe_allow_html=True)
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        st.markdown("</div>", unsafe_allow_html=True)

    # Try import Gemini functions
    has_gemini = import_util.find_spec("gemini") is not None
    chat_with_frostmart, generate_frostmart_report = None, None
    if has_gemini:
        try:
            from gemini import chat_with_frostmart, generate_frostmart_report
        except Exception:
            pass

    # User chat input
    if prompt := st.chat_input("Ask about sales trends, wastage, or performance..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        if chat_with_frostmart is None:
            response = "⚠️ Chat system not configured. Ensure gemini.py is available."
        else:
            with st.spinner("Analyzing FrostMart data..."):
                try:
                    response = chat_with_frostmart(prompt, "FrostMart knowledge base", st.session_state.chat_history)
                except Exception as e:
                    response = f"⚠️ Chat error: {e}"
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.chat_history.append({"user": prompt, "assistant": response})
        st.rerun()

    # Clear chat
    if st.button("🗑️ Clear Chat", key="clear_chat"):
        st.session_state.messages, st.session_state.chat_history = [], []
        st.rerun()

    # Generate full report
    st.markdown("---")
    if generate_frostmart_report is not None:
        if st.button("🧾 Generate Full FrostMart Analytics Report", use_container_width=True):
            with st.spinner("Generating comprehensive report..."):
                try:
                    report_text = generate_frostmart_report("FrostMart knowledge base and model insights")
                    st.success("✅ Report generated successfully!")
                    st.download_button("📥 Download Report as Text", report_text, "FrostMart_Report.txt")
                    st.text_area("📄 Report Preview", report_text[:3000], height=400)
                except Exception as e:
                    st.error(f"❌ Report generation failed: {e}")
    else:
        st.info("⚠️ Report generator unavailable. Add `gemini.py` with `generate_frostmart_report()` to enable this feature.")

# ===============================================================
# LEFT SIDE: MAIN APP CONTENT
# ===============================================================
with left_col:
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Introduction & Objectives",
        "🔮 Single Product Prediction",
        "📈 Batch Processing",
        "🧠 Model Details"
    ])

    # Tab 1
    with tab1:
        st.header("Project Overview")
        st.markdown("""
        FrostMart UK leverages AI-driven demand forecasting to reduce perishable waste and improve stock efficiency.
        This system predicts weekly sales for perishable SKUs, incorporating shelf life, weather, and marketing variables.
        """)
        st.subheader("Key Objectives")
        st.markdown("""
        - Forecast weekly demand  
        - Reduce overstock and waste  
        - Improve procurement accuracy  
        - Support data-driven decisions  
        """)
        st.info("Use the tabs above for single or batch predictions, and access detailed model metrics.")

    # Tab 2
    with tab2:
        st.header("🔮 Single Product Prediction")
        with st.form("perishable_prediction_form"):
            colA, colB = st.columns(2)
            with colA:
                product_id = st.number_input("Product ID", 1, 999999, 1)
                store_id = st.number_input("Store ID", 1, 9999, 1)
                product_name = st.text_input("Product Name", "Whole Wheat Bread 800g")
                category = st.selectbox("Product Category", ["Bakery", "Dairy", "Meat", "Beverages", "Produce"])
                shelf_life = st.number_input("Shelf Life (days)", 1, 60, 3)
                price = st.number_input("Price (£)", 0.0, 500.0, 2.5)
                cold_storage = st.number_input("Cold Storage Capacity", 0, 100000, 500)
                store_size = st.number_input("Store Size (sq ft)", 0, 50000, 1500)
            with colB:
                rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 20.5)
                avg_temp = st.number_input("Avg Temperature (°C)", -20.0, 50.0, 22.3)
                region = st.selectbox("Region", ["North", "South", "Midlands", "East", "West"])
                supplier = st.text_input("Supplier Name", "Fresh Foods Ltd")
                supply_capacity = st.number_input("Supply Capacity", 0, 1_000_000, 50000)
                marketing_spend = st.number_input("Marketing Spend (£)", 0.0, 100000.0, 500.0)
                month = st.number_input("Month", 1, 12, 1)
                day_of_week = st.number_input("Day of Week", 1, 7, 3)
                wastage = st.number_input("Wastage Units", 0, 100000, 100)
            submitted = st.form_submit_button("🔮 Predict Weekly Demand", use_container_width=True)

        if submitted:
            input_dict = {
                "Product_ID": product_id,
                "Store_ID": store_id,
                "Product_Name": product_name,
                "Product_Category": category,
                "Shelf_Life_Days": shelf_life,
                "Price": price,
                "Cold_Storage_Capacity": cold_storage,
                "Store_Size": store_size,
                "Rainfall": rainfall,
                "Avg_Temperature": avg_temp,
                "Region": region,
                "Supplier_Name": supplier,
                "Supply_Capacity": supply_capacity,
                "Marketing_Spend": marketing_spend,
                "Month": month,
                "Day_of_Week": day_of_week,
                "Wastage_Units": wastage
            }

            df = prepare_input_data(input_dict)
            prediction = make_prediction(df)
            if prediction is not None:
                st.subheader("📊 Prediction Results")
                st.metric("Predicted Weekly Units Sold", f"{prediction:.1f}")
                st.metric("Suggested Order (5% buffer)", f"{suggested_order(prediction)} units")

    # Tab 3
    with tab3:
        st.header("📈 Batch Processing")
        uploaded_file = st.file_uploader("Upload CSV file", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())
            if st.button("🚀 Predict Batch"):
                preds = []
                for _, row in df.iterrows():
                    df_row = prepare_input_data(row.to_dict())
                    preds.append(make_prediction(df_row))
                df["Predicted_Weekly_Units"] = preds
                df["Suggested_Order_5%"] = [suggested_order(p) for p in preds]
                st.dataframe(df)
                st.download_button("📥 Download Results", df.to_csv(index=False), "frostmart_predictions.csv", "text/csv")

    # Tab 4
    with tab4:
        st.header("🧠 Model Details & Performance")
        st.markdown("""
        - Final Model: Gradient Boosting Regressor (GBR)
        - Train R² = 0.87 | Test R² = 0.85
        - RMSE = 13.7 | MAE = 9.4
        """)
        st.info("Gradient Boosting was selected for its balance of performance and generalization.")
