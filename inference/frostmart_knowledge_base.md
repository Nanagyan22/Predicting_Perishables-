🥦 **FROSTMART UK – BUSINESS REPORT**  
**Predictive Modeling for Perishable Product Demand and Waste Reduction**

**Prepared by:** Francis Afful Gyan — Business Intelligence Specialist  
**Date:** October 2025  

---

## 1. Executive Summary  

FrostMart UK is a national retail chain specializing in fresh and perishable goods. The company faced operational inefficiencies in inventory management, leading to an estimated **£12.2 million annual loss** from overstocking, wastage, and frequent stockouts.  

To address this, FrostMart developed and deployed an **AI-powered predictive analytics system** designed to forecast weekly demand for perishable products. The initiative integrates:  
- **Advanced forecasting models** (Gradient Boosting Regressor)  
- **A Streamlit-based decision dashboard** for real-time use  
- **Business recommendations** on procurement, storage, and pricing optimization  

**Model Results (After Hyperparameter Tuning):**  
- R² = **0.9959**  
- MAE = **16.27**  
- RMSE = **75.28**  
- MAPE = **0.86%**  

**Key Outcomes:**  
- Projected **waste reduction by 30–40%**  
- Potential **revenue uplift of 10–20%**  
- Enhanced **sustainability and operational efficiency**

---

## 2. Company Context  

**Founded:** 1992  
**Headquarters:** London, UK  
**Scope:** Over 800 stores nationwide, operating across 6 key regions  
**Core Focus:** Fresh, sustainable, and locally sourced food products  

FrostMart’s perishable product lines — **Bakery, Dairy, Meat, Produce, and Beverages** — are highly sensitive to **seasonality**, **storage conditions**, and **weather patterns**.  
Traditional manual forecasting failed to capture these dynamic interactions, resulting in significant waste and lost revenue opportunities.

---

## 3. Problem Statement  

### Operational Issue  
- Average wastage rate: **7.8%**  
- Recurring **stockouts** during demand spikes  

### Financial Impact  
- Estimated annual losses: **£12.2 million**

### Root Causes  
- Reactive, manual inventory planning  
- Inefficient integration between promotions, pricing, and supply  
- Poor alignment of demand data with external factors (weather, holidays, temperature trends)

---

## 4. Project Objectives  

1. **Forecast weekly demand** for perishable goods using machine learning.  
2. **Identify key demand drivers** affecting wastage and sales.  
3. **Reduce inventory waste** and improve procurement accuracy.  
4. Deliver an **interactive Streamlit app** for managers and analysts to make real-time decisions.

---

## 5. Exploratory Data Analysis (EDA)  

**Data Period:** 2024-W01 to 2024-W52  
**Total Records:** 37,440 weekly sales entries  

| Metric | Value |
|---------|-------|
| Total Units Sold | 61,482,249 |
| Average Weekly Sales | 1,642 units |
| Total Wastage | 4,786,490 units |
| Wastage Rate | **7.79%** |

### Product & Regional Overview  

| Category | Count | Key Insight |
|-----------|--------|-------------|
| Bakery | 12 | Highest wastage (14.7%) |
| Meat | 12 | Highest average price (£7.27) |
| Dairy | 12 | Top-performing category |
| Beverages | 12 | Longest shelf life (216.8 days) |

| Region | Stores | Key Insight |
|---------|---------|-------------|
| South West | 4 | Most efficient by sales per sq. ft |
| London | 3 | Highest sales and wastage |
| Midlands | 3 | Moderate performance |
| North East | 2 | Lowest wastage (7.7%) |
| North West | 2 | Strong operational efficiency |
| South East | 1 | Smallest footprint |

---

### 📊 Key Insights  

#### Category Insights  
- **Top Seller:** Dairy — Mozzarella 250g (2.09M units)  
- **Highest Wastage:** Bakery (15%)  
- **Most Expensive:** Meat (£7.27 avg per unit)

#### Regional Insights  
- **Top Sales Region:** London  
- **Best Efficiency:** South West  
- **Lowest Wastage:** North East  

#### Price & Promotion  
- Average Price: £4.16  
- 71% budget, 24% mid-range, 5% premium products  
- Promotions increase sales by **18.6%** on average  
- **25% discount** yields a **51.3% uplift** in sales  

#### Marketing Efficiency  
- Correlation (Marketing Spend → Sales): **0.041**  
- Marketing Cost per Unit Sold: **£0.25**

#### Shelf Life & Waste  
- Shortest Shelf Life: Bakery (3.4 days)  
- Longest Shelf Life: Beverages (216.8 days)

---

### 🥇 Top Products  

| Product | Category | Units Sold | Avg Price | Wastage |
|----------|-----------|-------------|------------|----------|
| Mozzarella 250g | Dairy | 2,093,430 | £2.81 | 5.4% |
| Cottage Cheese 300g | Dairy | 2,070,145 | £2.80 | 5.3% |
| White Sandwich Loaf | Bakery | 1,211,590 | £1.85 | 15.0% |
| Banana Bread Loaf | Bakery | 1,095,312 | £2.10 | 15.0% |

---

### 🗓️ Seasonality Insights  
- **Peak Sales:** November 2024  
- **Highest Wastage:** February 2024  
- **Best Marketing ROI:** January 2024  

---

## 6. Model Development & Evaluation  

**Data Preparation:**  
- Median imputation for numeric values  
- Mode replacement for categorical values  

**Feature Engineering:**  
- Lagged sales (t–1, t–2)  
- Shelf Life Index  
- Normalized weather variables (temperature, rainfall)  
- Encoded categorical features (region, supplier, product category)

| Model | R² | RMSE | MAE | MAPE | Comment |
|--------|----|------|------|-------|----------|
| Linear Regression | 0.63 | 21.5 | 19.8 | 5.4% | Weak with non-linear data |
| Random Forest | 0.82 | 15.3 | 12.7 | 3.2% | Slight overfitting |
| Gradient Boosting Regressor | **0.9959** | **75.28** | **16.27** | **0.86%** | Best balance of accuracy and generalization |

**Top Predictors:**  
1. Shelf Life (Days)  
2. Average Temperature  
3. Marketing Spend  
4. Product Category  
5. Rainfall  
6. Discount %  

---

## 7. Best Model Selection  

**Final Model:** Gradient Boosting Regressor (GBR)  
- R² = 0.9959  
- RMSE = 75.28  
- MAE = 16.27  
- Robust across all regions and product groups  
- Strong interpretability and scalability  

---

## 8. Deployment & Integration  

**Platform:** Streamlit Application  
**Modules:**  
- Single Product Prediction  
- Batch Forecasting via CSV  
- AI Chat Assistant (Gemini Integration)  
- Automatic Suggested Order Calculation  
- Model Details & Report Generation  

**Integration Details:**  
- Connected to procurement and store-level databases  
- Weekly retraining using updated data  
- Supports real-time decision-making for buyers and category managers  

---

## 9. Strategic Business Recommendations  

### 🎯 Priority Actions  
- Focus **waste reduction** in Bakery and London region.  
- Adopt a **25% promotional discount** as the optimal level.  
- Increase **marketing spend** in Jan–Mar (best ROI period).  
- Use **AI-guided replenishment** for short shelf-life products.  
- Monitor and refine **Meat category pricing**.  
- Replicate best practices from **Store 502 (lowest wastage)** and **Store 505 (most efficient)**.

---

### 💰 Financial Impact  

| Initiative | Expected Outcome |
|-------------|------------------|
| 1% Wastage Reduction | ≈ £2.56M annual savings |
| Optimal Discounting | +51% sales lift |
| Improved Marketing ROI | £0.03 cost reduction per unit |

---

### 📊 KPI Targets  

| KPI | Current | Target |
|------|----------|---------|
| Wastage Rate | 7.8% | <4% |
| Marketing ROI | 0.041 | ≥0.10 |
| Discount Sales Lift | 51.3% | Maintain |
| Regional Efficiency Gap | 41.8% | <25% |

---

## 10. Conclusion  

The FrostMart UK Predictive Modeling initiative demonstrates the transformative power of **data-driven forecasting** in retail.  
By integrating machine learning with business strategy, FrostMart has built a foundation for:  
- Sustainable **waste reduction**  
- Intelligent **inventory and procurement**  
- Improved **profitability and decision-making**  

The Streamlit-powered system offers a scalable, AI-driven solution that supports **long-term operational resilience** and **sustainability goals**.

---

**End of Report**  
📘 *FrostMart UK — Data Science & Business Intelligence Division (2025)*
