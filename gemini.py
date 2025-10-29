import os
import google.genai as genai

from dotenv import load_dotenv

load_dotenv()

# GEMINI CLIENT SETUP
def get_client():
    """Get or create Gemini client"""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    return genai.Client(api_key=api_key)


# AI CHAT FUNCTION — FrostMart UK


def chat_with_frostmart(user_question: str, knowledge_base: str, chat_history: list = None) -> str:
    """
    Chat with AI using the FrostMart UK predictive analytics knowledge base.
    """
    system_prompt = f"""You are a Data Science and Business Analytics assistant for FrostMart UK — 
a national retail chain specializing in fresh and perishable goods. 
Use the knowledge base below to answer all questions with clarity, precision, 
and reference to data, metrics, and KPIs.

YOUR ROLE:
- Provide insights strictly from the FrostMart UK dataset and business report.
- Support responses with data-driven evidence (e.g., R², wastage %, revenue lift).
- Maintain a professional tone aligned with business intelligence reporting.
- Use numerical formatting (commas, decimals, £ signs, %).
- Do NOT fabricate information outside the knowledge base.
- If asked about unrelated topics, politely state you can only answer FrostMart UK-related questions.

RESPONSE GUIDELINES:
1. Begin with a direct, factual answer.
2. Support with exact metrics or KPIs from the data.
3. Use bullet points or numbered lists when appropriate.
4. Provide concise explanations or implications.
5. Keep responses under 10 sentences when possible.
6. Avoid markdown symbols like * or ** — use plain text.
7. Be clear, analytical, and data-focused.

EXAMPLE INTERACTIONS:

User: "Which product category has the highest wastage?"
Good Response: "Bakery products recorded the highest wastage at 15%, largely due to short shelf life averaging 3.4 days."

User: "Which model performed best for forecasting?"
Good Response: "The Gradient Boosting Regressor performed best with an R² of 0.87 and RMSE of 13.7, outperforming Random Forest and Linear Regression."

User: "What financial impact could a 1% waste reduction achieve?"
Good Response: "A 1% wastage reduction translates to approximately £2.56 million in annual savings."

KNOWLEDGE BASE:
{knowledge_base}

Answer only based on this information.
"""

    try:
        client = get_client()
        if chat_history is None:
            chat_history = []

        messages = [system_prompt]
        for msg in chat_history[-10:]:
            messages.append(msg)
        messages.append(f"User Question: {user_question}")

        full_prompt = "\n\n".join(messages)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt,
            config={
                "temperature": 0.3,
                "top_p": 0.9,
                "max_output_tokens": 1024,
            },
        )

        return response.text or "I couldn't generate a response. Please try again."

    except Exception as e:
        return f"Error: {str(e)}"



# COMPREHENSIVE BUSINESS INSIGHTS REPORT GENERATOR


def generate_frostmart_report(knowledge_base: str) -> str:
    """
    Generate a detailed, professional FrostMart UK predictive analytics report.
    """
    prompt = f"""You are a Senior Business Intelligence Analyst at FrostMart UK.
Generate a comprehensive predictive analytics and business performance report 
based on the FrostMart knowledge base below.

KNOWLEDGE BASE:
{knowledge_base}

Your report should follow this structure and tone:

---
# FROSTMART UK — BUSINESS ANALYTICS & DEMAND FORECASTING REPORT

Date: October 2025
Prepared by: Data Science & Business Analytics Division

---

## 1. Executive Summary
- Overview of FrostMart’s operational inefficiencies, waste issues, and strategic response.
- Highlight predictive model performance and key financial impacts.
- Emphasize benefits in waste reduction, profitability, and process efficiency.

## 2. Company Background
- Brief background of FrostMart UK and its operational scope.
- Challenges associated with perishable goods and inventory forecasting.

## 3. Problem Statement
- Detail core operational and financial issues.
- Provide metrics such as loss figures (£12.2 million), wastage %, and regional impacts.

## 4. Objectives
- Summarize main goals: demand prediction, waste reduction, optimization, and dashboarding.

## 5. Exploratory Data Analysis
- Include descriptive insights about product categories, regional performance, pricing, and promotions.
- Highlight wastage rate (7.79%), shelf life insights, and top-performing products.
- Show category-wise and region-wise summaries with clear numeric trends.

## 6. Model Development & Evaluation
- Summarize preprocessing steps and models tested.
- Report model results: R², RMSE, and reasons for model selection.
- Identify top predictors (shelf life, temperature, marketing spend, discount, etc.).

## 7. Best Model Selection
- Justify the choice of Gradient Boosting Regressor (GBR).
- Highlight interpretability, performance metrics, and cross-segment stability.

## 8. Deployment & Integration
- Describe the Streamlit dashboard, real-time prediction features, and integration with procurement systems.

## 9. Strategic Recommendations
- Highlight key actions:
  - Reduce wastage in Bakery and London.
  - Optimize promotions (25% discount benchmark).
  - Improve marketing ROI (≥0.10 target).
  - AI-driven inventory for short shelf-life goods.
- Quantify financial impacts (e.g., £2.56M savings per 1% waste reduction).

## 10. Conclusion
- Reinforce FrostMart’s transition toward data-driven retail operations.
- Emphasize long-term sustainability, scalability, and profitability of the solution.

FORMAT RULES:
- Use # and ## headings in markdown style.
- Avoid asterisks or markdown styling for emphasis.
- Use proper number formatting (commas, £, decimals).
- Keep language professional, concise, and factual.
"""

    try:
        client = get_client()
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt,
            config={
                "temperature": 0.4,
                "top_p": 0.9,
                "max_output_tokens": 8192,
            },
        )

        return response.text or "Unable to generate report."

    except Exception as e:
        return f"Error generating report: {str(e)}"


# ===============================================================
# EXAMPLE USAGE
# ===============================================================

if __name__ == "__main__":
    frostmart_knowledge_base = """🥦 FROSTMART UK – BUSINESS REPORT
Predictive Modeling for Perishable Product Demand and Waste Reduction

Prepared by: Francis Afful Gyan, Business Intelligence Specialist
Date: October 2025
... (your full FrostMart UK text here) ...
"""

    # Example: Chat mode
    question = "Which category had the highest wastage and why?"
    print(chat_with_frostmart(question, frostmart_knowledge_base))

    # Example: Generate full report
    report = generate_frostmart_report(frostmart_knowledge_base)
    print(report)
