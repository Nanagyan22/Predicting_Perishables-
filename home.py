# streamlit_app.py
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
    <p>Demand forecasting & waste reduction — actionable weekly predictions</p>
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
.stAlert > div {
    padding: 1rem;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Load Model + Assets
# -------------------------
INFERENCE_DIR = "inference"

@st.cache_resource
def load_assets(inference_dir: str = INFERENCE_DIR) -> Tuple[Any, List[str], Any]:
    """
    Load model, features JSON, and encoding_artifacts.pkl (robust to pickle/joblib/xgboost).
    """
    model_path = os.path.join(inference_dir, "xgboost_model.pkl")
    features_path = os.path.join(inference_dir, "xgboost_features.json")
    enc_path = os.path.join(inference_dir, "encoding_artifacts.pkl")

    # Model
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
            try:
                booster = xgb.Booster()
                booster.load_model(model_path)
                model = booster
            except Exception as e:
                st.error(f"❌ Could not load model: {e}")
                st.stop()

    # Features
    if not os.path.exists(features_path):
        st.error(f"❌ Missing file: {features_path}")
        st.stop()
    with open(features_path, "r") as f:
        feature_names = json.load(f)
    if not isinstance(feature_names, list):
        st.error("❌ xgboost_features.json must contain a JSON list")
        st.stop()

    # Encodings
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


# Helper Functions

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

    # Initialize chat session
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Chat display area (expanded height)
    chat_container = st.container()
    with chat_container:
        chat_box = st.container()
        chat_box.markdown(
            "<div style='height:700px; overflow-y:auto; border:1px solid #ddd; padding:10px; border-radius:10px;'>",
            unsafe_allow_html=True
        )
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        chat_box.markdown("</div>", unsafe_allow_html=True)

    # Import optional frostmart_ai chat
    from importlib import util as import_util
    has_chat_function = import_util.find_spec("frostmart_ai") is not None
    chat_with_frostmart = None
    knowledge_base = None
    if has_chat_function:
        try:
            from frostmart_ai import chat_with_frostmart, knowledge_base  # type: ignore
        except Exception:
            chat_with_frostmart = None

    # User input
    if prompt := st.chat_input("Ask about sales trends, wastage, discounts, or model results..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        if chat_with_frostmart is None:
            response = (
                "⚠️ FrostMart AI chat is not configured.\n\n"
                "Provide a function `chat_with_frostmart(prompt, knowledge_base, chat_history)` for full answers."
            )
        else:
            with st.spinner("Analyzing FrostMart data..."):
                try:
                    response = chat_with_frostmart(prompt, knowledge_base, st.session_state.chat_history)
                except Exception as e:
                    response = f"⚠️ Chat error: {e}"

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.chat_history.append({"user": prompt, "assistant": response})
        st.rerun()

    # Clear chat
    if st.button("🗑️ Clear Chat", key="clear_chat"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    # Example prompts
    with st.expander("💡 Sample Questions"):
        st.markdown("""
        - Which product category has the highest wastage rate?
        - What is the estimated financial loss from perishable waste?
        - Which region has the best sales-to-waste efficiency?
        - How do promotions affect sales performance?
        - What are the main predictors of product demand?
        - When does peak demand occur across categories?
        - How much can be saved if wastage drops by 1%?
        - What actions can reduce waste in Bakery products?
        - How does shelf life influence total wastage?
        """)

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

    # -------------------------
    # Tab 1
    # -------------------------
    with tab1:
        st.header("Project Overview")
        st.markdown("""
        - **Problem Statement:**  
        FrostMart, a leading grocery and retail chain, faces daily challenges balancing **product freshness**, **waste reduction**, and **stock availability**.
        This system leverages machine learning to forecast product demand for perishable items, ensuring optimal stock levels across stores.
        """)
        st.subheader("Key Business Drivers")
        st.markdown("""
        - **Reduce Waste:** Minimize expired goods  
        - **Protect Revenue:** Prevent stockouts  
        - **Improve Efficiency:** Optimize supply chain planning
        """)
        st.subheader("Objectives")
        st.markdown("""
        1. Forecast weekly units sold  
        2. Use weather, pricing, and shelf-life data  
        3. Provide single & batch forecasting tools
        """)
        st.info("""
        Tabs:  
        - *Single Product Prediction* → Forecast one SKU  
        - *Batch Processing* → Upload CSV for multiple SKUs  
        - *Model Details* → View model metrics & importance
        """)


    # Tab 2: Single Product Prediction

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
                st.metric("Suggested Order Quantity (5% buffer)", f"{suggested_order(prediction)} units")
                st.info(f"Recent reported wastage: **{wastage} units**")


    # Tab 3 and Tab 4 remain unchanged
  
    with tab3:
        st.header("📈 Batch Processing")
        st.markdown("Upload a CSV file with multiple products to get batch predictions.")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data:")
                st.dataframe(batch_df.head())

                if st.button("Predict Batch Sales"):
                    st.info("Processing batch predictions...")
                    results = []
                    for _, row in batch_df.iterrows():
                        input_dict = row.to_dict()
                        df = prepare_input_data(input_dict)
                        prediction = make_prediction(df)
                        results.append(prediction if prediction is not None else np.nan)
                    batch_df["Predicted_Weekly_Units_Sold"] = results
                    st.success("Batch predictions completed!")
                    st.dataframe(batch_df)

                    csv = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=csv,
                        file_name='batch_predictions.csv',
                        mime='text/csv',
                    )
            except Exception as e:
                st.error(f"❌ Error processing file: {e}")

    with tab4:
        st.header("🧠 Model Details & Performance")
        st.markdown("Details about the XGBoost model used for predictions.")
        st.subheader("Model Performance Metrics")
        st.markdown("""
        - **RMSE:** 12.5 units  
        - **MAE:** 8.3 units  
        - **R² Score:** 0.87
        """)
    st.info("The model was trained on historical sales data, weather patterns, and inventory metrics to optimize perishable goods forecasting.")
        
