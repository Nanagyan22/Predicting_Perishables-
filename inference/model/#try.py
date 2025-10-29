"""
import localstream as st
import requests

st.title("Perishable Goods Sales Prediction" )

with st.form("prediction forms"):
    wasteage_unit = st.number_input("Wastage Units", min_value=0, value=100)
    product_name = st.text_input("Product Name", "whole wheat bread 800g")
    proucct_categoty = st.selectbox("Product Category", ["Bakery", "Dairy", "Fruits", "Vegetables", "Meat", "Seafood", ])
    shelf_life_days = st.number_input("Shelf Life (days)", min_value=1, value=3)
    price = st.number_input("Price ($)", min_value=0.0, value=0.0)
    cold_storage_capacity = st.number_input("Cold Storage Capacity", min_value=0, value=500)
    store_size = st.number_input("Store Size (sq ft)", min_value=0, value=1500)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=20.5)
    avg_temperature = st.number_input("Average Temperature (°C)", min_value=-10.0, value=22.3)
    region = st.selectbox("Region", ["North", "South", "East", "West"])


    submitted = st.form_submit_button("Predict Sales")

if submitted:
    try:
        data ={
            "Wastage_Units": wasteage_unit,
            "Product_Name": product_name,
            "Product_Category": proucct_categoty,
            "Shelf_Life_Days": shelf_life_days,
            "Price": price,
            "Cold_Storage_Capacity": cold_storage_capacity,
            "Store_Size": store_size,
            "Rainfall": rainfall,
            "Avg_Temperature": avg_temperature,
            "Region": region
        }
        API_URL = "http://localhost:3000/predict"
        response = requests.post(API_URL, json={"data": data})

        if response.status_code == 200:
            st.write(f"Estimated Unit Sold: (response.predictions)")
        else:
            st.error(f"API Error {response.status_code}: {response.text}")
                

    except Exception as e:
        st.error(f"Error preparing data: {e}")

if __name__ == "__main__":
    pass
    
"""