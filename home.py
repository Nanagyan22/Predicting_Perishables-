# import libraries
import os
import pickle
import joblib
import json
from math import ceil
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from importlib import util as import_util
from dotenv import load_dotenv
import docx
from gemini import chat_with_frostmart, generate_frostmart_report, load_knowledge_base



# Load the .env file from the same folder as app.py
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("⚠️ GEMINI_API_KEY not found. Please check your .env file.")
else:
    print("✅ GEMINI_API_KEY loaded successfully!")

# Gemini / FrostMart chat integration
has_gemini = import_util.find_spec("gemini") is not None
if has_gemini:
    try:
        from gemini import chat_with_frostmart, generate_frostmart_report, load_knowledge_base
    except Exception:
        chat_with_frostmart = None
        generate_frostmart_report = None
        load_knowledge_base = None
else:
    chat_with_frostmart = None
    generate_frostmart_report = None
    load_knowledge_base = None

# App config & header
st.set_page_config(
    page_title="FrostMart UK Perishable Demand Prediction System",
    page_icon="🥕",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
<div class="main-header">
    <h1>🥕 FROSTMART UK</h1>
    <h3>Perishable Demand Prediction System</h3>
    <p>Demand forecasting · Waste reduction · Inventory optimization</p>
    <p style="font-size:14px; margin-top:10px;"></p>
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
""",
    unsafe_allow_html=True,
)

# Inference artifact paths
INFERENCE_DIR = "inference"
MODEL_FILENAME = "xgboost_model.pkl"
FEATURES_FILENAME = "xgboost_features.json"
ENCODINGS_FILENAME = "encoding_artifacts.pkl"

MODEL_PATH = os.path.join(INFERENCE_DIR, MODEL_FILENAME)
FEATURES_PATH = os.path.join(INFERENCE_DIR, FEATURES_FILENAME)
ENCODINGS_PATH = os.path.join(INFERENCE_DIR, ENCODINGS_FILENAME)

# Load model + artifacts
@st.cache_resource
def load_assets(inference_dir: str = INFERENCE_DIR) -> Tuple[Any, List[str], Any]:
    # model
    if not os.path.exists(MODEL_PATH):
        st.error(f" Model file not found: {MODEL_PATH}")
        st.stop()

    model = None
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    except Exception:
        try:
            model = joblib.load(MODEL_PATH)
        except Exception:
            try:
                booster = xgb.Booster()
                booster.load_model(MODEL_PATH)
                model = booster
            except Exception as e:
                st.error(f" Could not load model: {e}")
                st.stop()

    # features
    if not os.path.exists(FEATURES_PATH):
        st.error(f" Features file not found: {FEATURES_PATH}")
        st.stop()

    try:
        with open(FEATURES_PATH, "r", encoding="utf-8") as f:
            feature_names = json.load(f)
    except Exception as e:
        st.error(f" Could not read features JSON: {e}")
        st.stop()

    if not isinstance(feature_names, list):
        st.error(" xgboost_features.json must contain a list of feature names.")
        st.stop()

    # encodings
    encodings = None
    if os.path.exists(ENCODINGS_PATH):
        try:
            with open(ENCODINGS_PATH, "rb") as f:
                encodings = pickle.load(f)
        except Exception:
            try:
                encodings = joblib.load(ENCODINGS_PATH)
            except Exception:
                encodings = None

    return model, feature_names, encodings

model, feature_names, encodings = load_assets()

# Helper functions
def prepare_input_data(input_dict: Dict[str, Any]) -> pd.DataFrame:
    prepared = {feat: 0.0 for feat in feature_names}
    for k, v in input_dict.items():
        if k in prepared:
            prepared[k] = v
    return pd.DataFrame([prepared], columns=feature_names)

def encode_input_data(df: pd.DataFrame, encodings: Any = None) -> pd.DataFrame:
    df_encoded = df.copy()
    obj_cols = df_encoded.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in obj_cols:
        if encodings and isinstance(encodings, dict) and col in encodings:
            encoder = encodings[col]
            try:
                df_encoded[col] = encoder.transform(df_encoded[col].astype(str))
            except Exception:
                known = getattr(encoder, "classes_", None)
                if known is not None:
                    df_encoded[col] = df_encoded[col].apply(
                        lambda x: int(np.where(known == str(x))[0][0]) if str(x) in known else 0
                    )
                else:
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        else:
            le = LabelEncoder()
            try:
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            except Exception:
                df_encoded[col] = df_encoded[col].astype(str).map(lambda x: hash(x) % 10000)
    return df_encoded

def make_prediction(input_df: pd.DataFrame) -> float:
    try:
        df_encoded = encode_input_data(input_df, encodings)
        if isinstance(model, xgb.Booster):
            dmatrix = xgb.DMatrix(df_encoded.values, feature_names=list(df_encoded.columns))
            preds = model.predict(dmatrix)
        else:
            preds = model.predict(df_encoded)
        return float(np.array(preds)[0])
    except Exception as e:
        st.error(f" Prediction failed: {e}")
        return None

def suggested_order(predicted_units: float, buffer_pct: float = 0.05) -> int:
    try:
        return int(ceil(predicted_units * (1 + float(buffer_pct))))
    except Exception:
        return 0

# Load DOCX Knowledge Base

kb_path = os.path.join("inference", "frostmart_knowledge_base.docx")
frost_kb = ""
if os.path.exists(kb_path):
    try:
        doc = docx.Document(kb_path)
        frost_kb = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    except Exception as e:
        st.warning(f"⚠️ Could not load FrostMart knowledge base: {e}")
else:
    st.warning("⚠️ FrostMart knowledge base file not found in /inference directory.")

# Layout columns: left main content, right chat assistant
left_col, chat_col = st.columns([3, 1])


# RIGHT: Gemini Chat Assistant
# FrostMart UK AI Assistant
with chat_col:
    st.markdown("### 🤖 AI Assistant")
    st.markdown("*Ask questions about the dataset or FrostMart operations*")

    # Initialize session states
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Reserved container for chat
    chat_height = 700
    chat_box = st.empty()

    # Render chat messages
    def render_chat():
        # Start scrollable chat container
        chat_container = f"""
        <div style='height:{chat_height}px; overflow-y:auto; overflow-x:hidden; padding:10px; border:1px solid #ddd; border-radius:10px; background-color:#f9f9f9; word-wrap:break-word;'>
        """
        chat_content = ""
        for msg in st.session_state.messages:
            if msg["role"] == "assistant":
                color = "#0b6b3a"  # green
                prefix = "🤖 Assistant:"
                align = "left"
            else:
                color = "#2a2a2a"  # dark gray
                prefix = "🧑 You:"
                align = "right"

            # Plain text output inside the container
            chat_content += f"<div style='text-align:{align}; color:{color}; margin:5px 0;'>{prefix} {msg['content']}</div>"

        chat_container += chat_content
        chat_container += "</div>"
        chat_box.markdown(chat_container, unsafe_allow_html=True)

    # Initial render
    render_chat()

    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        if not GEMINI_API_KEY or chat_with_frostmart is None:
            response = "⚠️ Chat assistant unavailable. Set GEMINI_API_KEY and load gemini.py correctly."
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        else:
            with st.spinner("Thinking..."):
                try:
                    response = chat_with_frostmart(prompt, frost_kb, st.session_state.chat_history)
                    st.session_state.chat_history.append({"role": "user", "content": prompt})
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                except Exception as e:
                    response = f"⚠️ Chat error: {e}"

        st.session_state.messages.append({"role": "assistant", "content": response})
        render_chat()
        st.rerun()

    # Clear chat
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        render_chat()
        st.rerun()

       # Sample questions outside the box
    with st.expander("💡 Sample Questions"):
        st.markdown("""
        - What is the total estimated annual loss from wastage and overstocking?
        - Which product categories have the highest wastage rates?
        - Which regions perform best in terms of sales and efficiency?
        - How much improvement in profitability is expected after deploying the AI model?
        - What are the performance metrics (R², RMSE, MAPE) of the Gradient Boosting model?
        - How does the AI model help reduce waste and optimize ordering?
        - What business recommendations does the AI provide for FrostMart UK?
        - What are the main modules of the Streamlit AI system and their functions?
        """)


# FrostMart Report Generation
st.markdown("---")
if generate_frostmart_report is not None:
    st.subheader("🧾 Generate Full FrostMart Analytics Report")
    st.markdown("""
    This feature compiles key insights from FrostMart UK’s predictive demand model and 
    knowledge base — summarizing trends in sales, wastage, supply chain performance, 
    and forecasting accuracy across stores and product categories.
    """)
    kb_path = os.path.join("inference", "frostmart_knowledge_base.md")
    frost_kb = ""
    if os.path.exists(kb_path):
        try:
            with open(kb_path, "r", encoding="utf-8") as f:
                frost_kb = f.read()
        except Exception as e:
            st.warning(f"⚠️ Could not load FrostMart knowledge base: {e}")
    else:
        st.warning("⚠️ FrostMart knowledge base file not found in /inference directory.")

    if st.button("🚀 Click Here to Generate Full Report", use_container_width=True, type="primary"):
        with st.spinner("Generating comprehensive FrostMart UK analytics report..."):
            try:
                if not frost_kb:
                    raise ValueError("Knowledge base is empty or could not be loaded.")

                report_text = generate_frostmart_report(frost_kb)

                st.success("✅ Report generated successfully!")

                st.download_button(
                    label="📥 Download Report as Text",
                    data=report_text,
                    file_name="FrostMart_UK_Analytics_Report.txt",
                    mime="text/plain"
                )

                st.text_area("📄 Report Preview", report_text[:3000], height=400)

            except Exception as e:
                st.error(f"❌ Report generation failed: {e}")
                st.info("💡 Ensure that `gemini.py` is configured correctly and the knowledge base file is present.")
else:
    st.info("⚠️ Report generator unavailable. Please add `gemini.py` with the function `generate_frostmart_report()` to enable this feature.")


# LEFT: Main App Tabs
with left_col:
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Introduction & Objectives", "🔮 Single Product Prediction", "📈 Batch Processing", "🧠 Model Details"]
    )

# TAB 1
with tab1:
    st.header("Project Overview: Predicting Demand for Perishable Goods at FrostMart UK")
    st.markdown("""
    - **Problem Statement:**  
    FrostMart UK, a national grocery chain with over 800 stores, faces a persistent challenge in managing perishable goods efficiently.  
    Products such as fresh produce, bakery items, and chilled foods generate a significant share of revenue but also contribute heavily to waste due to short shelf lives, weather fluctuations, and unpredictable customer demand.

    This application leverages a **machine learning–based demand forecasting model** to predict weekly product demand across stores, 
    helping the company minimize waste, improve inventory turnover, and ensure product availability.
    """)
    st.subheader("Key Business Drivers")
    st.markdown("""
    - **Waste Reduction:** Minimize overstocking and markdown losses, improving sustainability and profit margins.  
    - **Procurement Accuracy:** Align supplier orders and deliveries with store-level demand patterns.  
    - **Operational Efficiency:** Optimize cold storage usage and replenishment scheduling.  
    - **Revenue Maximization:** Prevent lost sales due to understocking during demand spikes.  
    - **Sustainability Commitment:** Support FrostMart UK’s corporate goals of reducing food waste and carbon footprint.
    """)
    st.subheader("Project Objectives")
    st.markdown("""
    1. **Develop a Predictive Model:** Build a machine learning model to forecast weekly perishable demand at the store and product levels.  
    2. **Integrate Multi-Factor Data:** Incorporate variables such as weather, shelf life, store characteristics, supplier capacity, and marketing spend.  
    3. **Deliver Actionable Insights:** Provide automated order recommendations with safety buffers to aid procurement teams.  
    4. **Improve Decision Support:** Enable planners and managers to visualize demand forecasts and make data-driven restocking decisions.  
    5. **Enhance Sustainability:** Reduce waste, improve stock rotation, and strengthen FrostMart UK’s commitment to environmental responsibility.
    """)
    st.subheader("Expected Outcomes")
    st.markdown("""
    - **Operational Impact:** Substantial reduction in perishable waste and markdown losses across store networks.  
    - **Financial Benefit:** Improved revenue through accurate demand prediction and optimized procurement.  
    - **Strategic Advantage:** Data-driven replenishment recommendations and enhanced supplier coordination.  
    - **Scalability:** A modular framework that can extend forecasting to multiple product categories and international markets.
    """)
    st.subheader("How to Use This App")
    st.info("""
    1. **Single Product Prediction:** Go to the *'Single Product Prediction'* tab, enter product and store details, and get an instant weekly demand forecast.  
    2. **Batch Processing:** Use the *'Batch Processing'* tab to upload a CSV file of multiple products for large-scale forecasting and order planning.  
    3. **Model Details:** Review model performance, accuracy metrics, and feature importance in the *'Model Details'* tab.  
    """)
# TAB 2: Single Product Prediction
with tab2:
    st.header("🔮 Single Product Prediction")
    with st.form("perishable_prediction_form"):
        colA, colB = st.columns(2)
        with colA:
            product_id = st.number_input("Product ID", min_value=1, max_value=999999, value=1, step=1)
            store_id = st.number_input("Store ID", min_value=1, max_value=9999, value=1, step=1)
            product_name = st.text_input("Product Name", value="Whole Wheat Bread 800g")
            product_category = st.selectbox("Product Category", ["Bakery", "Dairy", "Meat", "Beverages", "Produce"])
            shelf_life_days = st.number_input("Shelf Life (days)", min_value=1, max_value=180, value=3)
            price = st.number_input("Price (£)", min_value=0.0, max_value=1000.0, value=2.5, step=0.01)
            cold_storage_capacity = st.number_input("Cold Storage Capacity", min_value=0, max_value=1_000_000, value=500)
            store_size_sqft = st.number_input("Store Size (sq ft)", min_value=0, max_value=100_000, value=1500)
        with colB:
            rainfall_mm = st.number_input("Rainfall (mm)", min_value=0.0, max_value=1000.0, value=20.5, step=0.1)
            avg_temp_c = st.number_input("Avg Temperature (°C)", min_value=-20.0, max_value=50.0, value=22.3, step=0.1)
            region = st.selectbox("Region", ["North", "South", "Midlands", "East", "West"])
            supplier_name = st.text_input("Supplier Name", value="Fresh Foods Ltd")
            supply_capacity = st.number_input("Supply Capacity (units/week)", min_value=0, max_value=10_000_000, value=50000)
            marketing_spend = st.number_input("Marketing Spend (£)", min_value=0.0, max_value=1_000_000.0, value=500.0)
            month = st.selectbox("Month", list(range(1, 13)), index=0)
            day_of_week = st.selectbox("Day of Week (1=Mon)", list(range(1, 8)), index=2)
            wastage_units = st.number_input("Historic Wastage Units (last period)", min_value=0, max_value=1_000_000, value=100)

        submitted = st.form_submit_button("🔮 Predict Weekly Demand", use_container_width=True)

    if submitted:
        input_dict = {
            "Product_ID": product_id,
            "Store_ID": store_id,
            "Product_Name": product_name,
            "Product_Category": product_category,
            "Shelf_Life_Days": shelf_life_days,
            "Price": price,
            "Cold_Storage_Capacity": cold_storage_capacity,
            "Store_Size_SqFt": store_size_sqft,
            "Rainfall_mm": rainfall_mm,
            "Avg_Temperature_C": avg_temp_c,
            "Region": region,
            "Supplier_Name": supplier_name,
            "Supply_Capacity": supply_capacity,
            "Marketing_Spend": marketing_spend,
            "Month": month,
            "Day_of_Week": day_of_week,
            "Wastage_Units": wastage_units,
        }

        df_input = prepare_input_data(input_dict)
        predicted_units = make_prediction(df_input)

        if predicted_units is not None:
            suggested = suggested_order(predicted_units)
            st.subheader("📊 Prediction Results")
            st.metric("Predicted Weekly Units Sold", f"{predicted_units:.1f}")
            st.metric("Suggested Order (5% buffer)", f"{suggested} units")

            st.markdown("---")
            st.write(
                f"📝 The total **predicted weekly units sold** for **{product_name}** (Store {store_id}) "
                f"is approximately **{predicted_units:.1f} units**. "
                f"This forecast uses historical sales, weather, promotions, and product/supplier features to estimate demand."
            )
            st.write(
                f"📦 The suggested order quantity includes a **5% safety buffer** to protect against unexpected demand variability "
                f"and minor supply delays resulting in **{suggested} units** recommended for procurement."
            )

            if predicted_units < 50:
                st.info("📉 Low predicted demand — consider reducing order size, using targeted promotions, or limiting shelf stocking.")
            elif predicted_units > supply_capacity:
                st.warning("⚠️ Predicted demand exceeds supplier supply capacity. Consider supplier coordination or redistribution across stores.")
            elif predicted_units > 500:
                st.warning("🚀 High predicted demand — ensure logistics capacity (cold-chain) and supplier lead-times are aligned.")


# TAB 3: Batch Processing
with tab3:
    st.header("📈 Batch Perishable Demand Forecasting")
    st.markdown("**Upload a CSV file with product-level data for batch forecasting.**")

    sample_row = {
        "Product_ID": 101,
        "Store_ID": 5,
        "Product_Name": "Fresh Strawberries 250g",
        "Product_Category": "Produce",
        "Shelf_Life_Days": 5,
        "Price": 3.49,
        "Cold_Storage_Capacity": 5000,
        "Store_Size_SqFt": 2000,
        "Rainfall_mm": 15.2,
        "Avg_Temperature_C": 21.5,
        "Region": "South",
        "Supplier_Name": "BerryCo Ltd",
        "Supply_Capacity": 12000,
        "Marketing_Spend": 450.0,
        "Month": 6,
        "Day_of_Week": 4,
        "Wastage_Units": 85
    }
    template_df = pd.DataFrame([sample_row])

    st.download_button(
        "📥 Download CSV Template",
        template_df.to_csv(index=False),
        "perishable_demand_template.csv",
        "text/csv",
        help="Download a template CSV file with sample data format"
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", help="Upload your perishable product data file")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(df)} product records.")
            st.subheader("📊 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Rows", len(df))
            with col2:
                st.metric("Total Columns", len(df.columns))

            missing_cols = [col for col in feature_names if col not in df.columns]
            if missing_cols:
                st.warning(f"⚠️ Missing columns detected: {missing_cols}")
                st.info("Missing columns will be automatically filled with default values (0 or 0.0).")

            if st.button("🚀 Process Batch", use_container_width=True, type="primary"):
                with st.spinner("Processing batch... This may take a moment."):
                    try:
                        process_df = df.copy()
                        processed_data = []
                        for idx, row in process_df.iterrows():
                            input_dict = {}
                            for feature in feature_names:
                                input_dict[feature] = row[feature] if feature in row else 0
                            processed_data.append(input_dict)

                        batch_df = pd.DataFrame(processed_data, columns=feature_names)

                        preds = []
                        for _, row in batch_df.iterrows():
                            df_row = prepare_input_data(row.to_dict())
                            preds.append(make_prediction(df_row))

                        batch_df["Predicted_Weekly_Units"] = preds
                        batch_df["Suggested_Order_5pct"] = batch_df["Predicted_Weekly_Units"].apply(lambda x: suggested_order(x if x else 0))

                        results_df = df.copy()
                        results_df["Predicted_Weekly_Units"] = batch_df["Predicted_Weekly_Units"]
                        results_df["Suggested_Order_5pct"] = batch_df["Suggested_Order_5pct"]

                        def get_recommendation(units):
                            if units >= 1000:
                                return "High demand expected — ensure supplier coordination."
                            elif units >= 300:
                                return "Moderate demand — maintain regular replenishment schedule."
                            elif units >= 50:
                                return "Low demand — adjust inventory levels to avoid overstocking."
                            else:
                                return "Very low demand — consider limited procurement or bundle promotions."

                        results_df["Recommendation"] = results_df["Predicted_Weekly_Units"].apply(get_recommendation)

                        st.subheader("📈 Batch Forecasting Results")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Products Forecasted", len(results_df))
                        with col2:
                            high_demand = (results_df["Predicted_Weekly_Units"] >= 1000).sum()
                            st.metric("High Demand Products", high_demand)
                        with col3:
                            avg_forecast = np.nanmean(results_df["Predicted_Weekly_Units"])
                            st.metric("Average Predicted Demand", f"{avg_forecast:.1f} units")

                        st.subheader("📊 Detailed Forecast Results")
                        key_cols = ["Product_Name", "Product_Category", "Predicted_Weekly_Units", "Suggested_Order_5pct", "Recommendation"]
                        show_cols = [col for col in key_cols if col in results_df.columns]
                        st.dataframe(results_df[show_cols + [c for c in df.columns if c not in show_cols][:5]], use_container_width=True)

                        csv_data = results_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Full Results as CSV",
                            csv_data,
                            "frostmart_batch_forecast_results.csv",
                            "text/csv",
                            help="Download complete results with demand forecasts and recommendations"
                        )

                    except Exception as e:
                        st.error(f"❌ Error during batch processing: {str(e)}")
                        st.info("💡 Please verify your CSV file format matches the expected structure.")

        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")
            st.info("💡 Please ensure the uploaded file is a valid CSV format.")


# TAB 4: Model Details & Performance
with tab4:
    st.header("🧠 Model Details & Performance")
    st.markdown(f"**Model file:** `{MODEL_PATH}`  \n**Feature count:** {len(feature_names)}  ")
    st.subheader("📊 Test Performance After Hyperparameter Tuning")
    st.metric("R² Score", "0.9959")
    st.metric("Mean Absolute Error (MAE)", "16.27")
    st.metric("Root Mean Squared Error (RMSE)", "75.28")
    st.metric("Mean Absolute Percentage Error (MAPE)", "0.86%")
    st.info("The tuned XGBoost model provides very high explained variance (R² ~ 0.9959). Use domain checks (extreme promotion weeks, supply disruptions) before acting on single predictions.")

    st.subheader("Feature Importance")
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:20]
            imp_df = pd.DataFrame(feat_imp, columns=["feature", "importance"])
            st.dataframe(imp_df)
        elif isinstance(model, xgb.Booster):
            fmap = model.get_score(importance_type="weight")
            items = sorted(fmap.items(), key=lambda x: x[1], reverse=True)[:20]
            imp_df = pd.DataFrame(items, columns=["feature", "importance"])
            st.dataframe(imp_df)
        else:
            st.info("Feature importance not available for this model type.")
    except Exception as e:
        st.error(f"Could not compute feature importances: {e}")

    st.markdown("---")
    st.subheader("Business Impact & Next Steps")
    st.markdown(
        """
- Use the predictions to create weekly procurement plans.
- Continuously collect sales and wastage data to retrain and improve model accuracy.
- Integrate model outputs into FrostMart’s inventory management system for automated restocking alerts.
- Monitor forecast accuracy and adjust safety buffers according to product perishability and lead times.
- Consider seasonal trends, promotions, and supply constraints when finalizing orders.
"""
    )
