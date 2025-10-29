import localstream as st
import requests

# -------------------------
# API CONFIGURATION
# -------------------------
API_URL = "http://127.0.0.1:3000/predict"  # adjust if deployed remotely

st.set_page_config(
    page_title="Perishable Goods Sales Predictor",
    page_icon="🛒",
    layout="centered"
)

st.title("🛍️ Perishable Goods Sales Prediction")
st.markdown("Enter product and store details below to predict sales.")

# -------------------------
# INPUT FORM
# -------------------------
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        product_id = st.number_input("Product ID", min_value=1, value=1)
        store_id = st.number_input("Store ID", min_value=1, value=1)
        product_name = st.text_input("Product Name", "Whole Wheat Bread 800g")
        product_category = st.text_input("Product Category", "Bakery")
        shelf_life_days = st.number_input("Shelf Life (days)", min_value=1, value=3)
        price = st.number_input("Price ($)", min_value=0.0, value=2.5)
        cold_storage = st.number_input("Cold Storage Capacity", min_value=0, value=500)
        store_size = st.number_input("Store Size (sq ft)", min_value=0, value=1500)

    with col2:
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=20.5)
        avg_temp = st.number_input("Average Temperature (°C)", min_value=-10.0, value=22.3)
        region = st.text_input("Region", "North")
        supplier = st.text_input("Supplier Name", "Fresh Foods Ltd")
        supply_capacity = st.number_input("Supply Capacity", min_value=0, value=50000)
        marketing_spend = st.number_input("Marketing Spend ($)", min_value=0.0, value=500.0)
        month = st.number_input("Month", min_value=1, max_value=12, value=1)
        day_of_week = st.number_input("Day of Week", min_value=1, max_value=7, value=3)
        wastage_units = st.number_input("Wastage Units", min_value=0, value=100)

    submitted = st.form_submit_button("Predict Sales 🚀")

# -------------------------
# SEND REQUEST TO API
# -------------------------
if submitted:
    st.write("⏳ Sending data to FastAPI for prediction...")

    payload = {
        "data": {
            "Wastage_Units": wastage_units,
            "Product_Name": product_name,
            "Product_Category": product_category,
            "Shelf_Life_Days": shelf_life_days,
            "Price": price,
            "Cold_Storage_Capacity": cold_storage,
            "Store_Size": store_size,
            "Rainfall": rainfall,
            "Avg_Temperature": avg_temp,
            "Region": region
        }
    }

    try:
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()
            st.success(f" **Predicted Sales:** {result['predicted_sales']:.2f}")
        else:
            st.error(f" API Error {response.status_code}: {response.text}")

    except requests.exceptions.ConnectionError:
        st.error(" Could not connect to FastAPI. Make sure the API is running on port 3000.")
