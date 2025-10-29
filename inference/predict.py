from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import pickle
import joblib
import json
import os
from typing import Any, Dict
import uvicorn
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Perishable Goods Sales Prediction API", version="1.0")

class Item(BaseModel):
    data: Dict[str, Any] = Field(
        ...,
        example={
            "Wastage_Units": 100,
            "Product_Name": "Whole Wheat Bread 800g",
            "Product_Category": "Bakery",
            "Shelf_Life_Days": 3,
            "Price": 2.5,
            "Cold_Storage_Capacity": 500,
            "Store_Size": 1500,
            "Rainfall": 20.5,
            "Avg_Temperature": 22.3,
            "Region": "North"
        }
    )


@app.post("/predict")
def predict_sales(item: Item):
    """
    Expects JSON like:
    {
      "data": { ...fields above... }
    }
    """
    try:
        base_dir = os.path.dirname(__file__)  
        
        model_path = os.path.join(base_dir, "model", "xgboost_model.pkl")
        enc_path = os.path.join(base_dir, "encoding_artifacts.pkl")
        features_path = os.path.join(base_dir, "xgboost_features.json")

        # Load model: 
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
            except Exception as ex_joblib:
                # fallback to pickle if joblib fails for some reason
                with open(model_path, "rb") as model_file:
                    model = pickle.load(model_file)
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # --- Load encoding artifacts ---
        if os.path.exists(enc_path):
            # Try joblib first, then pickle (covers both saving approaches)
            try:
                encoding_artifacts = joblib.load(enc_path)
            except Exception:
                with open(enc_path, "rb") as enc_file:
                    encoding_artifacts = pickle.load(enc_file)
        else:
            raise FileNotFoundError(f"Encoding artifacts not found: {enc_path}")

        # --- Load feature names (JSON) ---
        if os.path.exists(features_path):
            with open(features_path, "r") as feat_file:
                feature_names = json.load(feat_file)
        else:
            raise FileNotFoundError(f"Feature names JSON not found: {features_path}")

        # --- Prepare input ---
        input_data = pd.DataFrame([item.data])

        # --- Apply label encoders (if present) ---
        # Expect encoding_artifacts to contain a dict: {"label_encoders": {"col": LabelEncoder, ...}, "low_cardinality_cols": [...]}
        label_encoders = encoding_artifacts.get("label_encoders", {})
        for col, le in label_encoders.items():
            if col in input_data.columns:
                # if value unseen, set to -1; if seen, transform
                input_data[col] = input_data[col].apply(
                    lambda x: int(le.transform([x])[0]) if x in getattr(le, "classes_", []) else -1
                )

        # --- Low-cardinality columns stored as categories (if any) ---
        for col in encoding_artifacts.get("low_cardinality_cols", []):
            if col in input_data.columns:
                # .cat.codes requires dtype category
                input_data[col] = input_data[col].astype("category").cat.codes

        # --- Align input to model features (fill missing with 0) ---
        input_data = input_data.reindex(columns=feature_names, fill_value=0)

        # --- Predict ---
        prediction = model.predict(input_data)

        return {"predicted_sales": float(prediction[0])}

    except FileNotFoundError as fnf:
        # more descriptive 404-ish error
        raise HTTPException(status_code=500, detail=str(fnf))
    except Exception as e:
        # any other error
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # default port comes from env PORT or fallback to 3000
    port = int(os.getenv("PORT", 3000))
    print(f"🚀 Starting FastAPI server on port {port} ...")
    # use module path so reload works with -m
    uvicorn.run("inference.predict:app", host="0.0.0.0", port=port, reload=True)
