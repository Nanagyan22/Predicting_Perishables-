# 🤖 FrostMart AI Chat Assistant

Powered by **Gemini 2.5 Flash**, this interactive AI assistant helps you ask **data-driven questions** directly about FrostMart’s predictive analytics and performance metrics.

### Example Questions:
- Which product category has the highest wastage?
- How does temperature affect Bakery sales?
- What’s the estimated savings from a 1% waste reduction?
- Which region achieved the highest sales efficiency?
- What are the top predictors of demand?

🗨️ *The chat window is located on the right-hand side of the Streamlit app.*

---

# 📄 Generate Full Business Report

Click the **“📄 Generate Full FrostMart Report”** button inside the app to automatically generate a **comprehensive business analytics report** powered by **Gemini 2.5 Pro**.

Each report is created dynamically from FrostMart’s internal knowledge base and includes:

- Executive Summary  
- Model Evaluation & Insights  
- Business Intelligence Findings  
- Financial Impact & Recommendations  

📊 The generated report provides structured analysis on performance, wastage reduction potential, and predictive accuracy.

---

# 🧾 Model Summary

| Model | Train R² | Test R² | RMSE | MAE |
|--------|-----------|---------|------|------|
| Linear Regression | 0.65 | 0.60 | 15.8 | 10.2 |
| Random Forest | 0.82 | 0.80 | 13.5 | 9.6 |
| **Gradient Boosting Regressor (Final)** | **0.87** | **0.85** | **13.7** | **9.4** |

---

# 🔑 Top Predictors

| Feature | Importance (%) |
|----------|----------------|
| Marketing Spend | 21.3 |
| Avg Temperature | 18.9 |
| Shelf Life Days | 13.5 |
| Region | 10.7 |
| Rainfall | 9.2 |

These features contributed most significantly to FrostMart’s weekly demand forecasting accuracy across regional and product-level models.

---

# ☁️ Deploying on Streamlit Cloud

To deploy **FrostMart Perishable Demand Prediction System** on **Streamlit Cloud**:

1. Upload your project folder to **GitHub**.  
2. Visit [https://perishables.streamlit.app/](https://perishables.streamlit.app/) → click **New app**.  
3. Select your repository and choose `streamlit_app.py` as the **entry file**.  
4. Go to **App Settings → Secrets** and add your Gemini API key:




---

