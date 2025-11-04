# frostmart_gemini.py — FrostMart UK Predictive Analytics Assistant

import os
from dotenv import load_dotenv

# Load .env explicitly
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

# Get API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError(
        "⚠️ GEMINI_API_KEY not found in environment variables. "
        "Please create a .env file with GEMINI_API_KEY=your_key_here"
    )

# Optional: Google Gemini AI
try:
    import google.genai as genai
except ModuleNotFoundError:
    genai = None

# DOCX reader
try:
    import docx
except ModuleNotFoundError:
    docx = None


# ---------------------------
# Load Knowledge Base
# ---------------------------
def load_knowledge_base(path: str) -> str:
    """Load the FrostMart knowledge base from a DOCX file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Knowledge base file not found: {path}")
    if docx is None:
        raise ModuleNotFoundError("python-docx module not installed. Install via 'pip install python-docx'.")
    
    document = docx.Document(path)
    full_text = "\n".join([para.text for para in document.paragraphs])
    return full_text


# ---------------------------
# Gemini Client
# ---------------------------
def get_client():
    """Create Gemini AI client."""
    if genai is None:
        raise ModuleNotFoundError(
            "google.genai module not installed. Install via 'pip install google-genai'."
        )
    return genai.Client(api_key=GEMINI_API_KEY)


# ---------------------------
# Chat with FrostMart AI
# ---------------------------
def chat_with_frostmart(user_question: str, knowledge_base: str, chat_history: list = None) -> str:
    """Chat with AI using the FrostMart UK knowledge base."""
    if genai is None:
        return "⚠️ google.genai module not installed. Cannot chat."

    if chat_history is None:
        chat_history = []

    system_prompt = f"""
You are an expert AI assistant for FrostMart UK — a national retail chain specializing in perishable goods.

YOUR ROLE:
- Answer questions using ONLY the knowledge base and dataset provided below
- Provide clear, specific, and data-driven responses
- Use exact numbers and metrics from the data
- Format your answers in a professional, easy-to-read manner
- If a question is outside the knowledge base, politely explain you can only answer questions about FrostMart UK data

RESPONSE GUIDELINES:
1. Start with a direct answer to the question
2. Support your answer with specific numbers and facts from the data
3. Use bullet points (with hyphens -) when presenting multiple items
4. Add brief context or insights when relevant
5. Keep responses concise but comprehensive
6. Do NOT use asterisks for bold or emphasis - write in plain text
7. Format numbers clearly with pound signs and commas (e.g., £4,100,000)
8. Keep your writing simple and readable
9. Always separate numbers from words with a space
10.Format numbers with commas for thousands and two decimals for cents
11.Example: "£57,570.00 and Wastage at £54,223.00"

KNOWLEDGE BASE:
{knowledge_base}
"""
    try:
        client = get_client()
        messages = [system_prompt] + [m["content"] for m in chat_history[-10:]] + [f"User Question: {user_question}"]
        full_prompt = "\n\n".join(messages)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt,
            config={"temperature": 0.3, "max_output_tokens": 1024},
        )

        return response.text if response.text else "No response generated."
    except Exception as e:
        return f"⚠️ Chat error: {str(e)}"


# ---------------------------
# Generate FrostMart Analytics Report
# ---------------------------
def generate_frostmart_report(knowledge_base: str) -> str:
    """Generate a detailed FrostMart UK business analytics report."""
    if genai is None:
        return "⚠️ google.genai module not installed. Cannot generate report."

    prompt = f"""
You are Francis Afful Gyan, a Business Intelligence Specialist for FrostMart UK.
Generate a comprehensive business analytics report using the knowledge base below.
Include insights on sales forecasting, wastage reduction, and operational optimization.

KNOWLEDGE BASE:
{knowledge_base}

Create a structured report with headings, numeric metrics, and clear recommendations covering:

---
# FROSTMART UK BUSINESS INSIGHTS REPORT

Date: October 2025
Prepared by: Francis Afful Gyan, Business Intelligence Specialist

---

## 1. Executive Summary
- Overview of current operational and financial state
- Key metrics: total revenue, wastage, and profit margins
- Critical findings

## 2. Financial Performance Analysis
- Annual estimated losses from waste & overstocking (£12.2M)
- Potential revenue uplift (10–20%)
- Profit margin analysis

## 3. Product & Regional Insights
- Top-selling and highest-wastage products
- Regional efficiency comparisons

## 4. Sales, Wastage & Pricing Insights
- Peak sales months and promotions impact
- Average prices per product category
- Marketing efficiency metrics

## 5. Seasonality & Trends
- Monthly patterns for sales, wastage, and ROI

## 6. Model Development & Evaluation
- Predictive model description (Linear Regression, Random Forest, Gradient Boosting)
- Key predictors (Shelf Life, Temperature, Marketing Spend, Product Category, Rainfall, Discount %)
- Model performance (R², RMSE, MAE, MAPE)

## 7. Deployment & Integration
- Streamlit app modules (single product prediction, batch CSV forecasting, AI chat assistant, suggested orders)
- Integration with procurement and store databases
- Weekly retraining procedures

## 8. Business Recommendations
- Waste reduction focus areas (Bakery, London)
- Promotion strategy (25% discount)
- Marketing ROI improvement (Jan–Mar)
- AI-guided replenishment for short shelf-life items
- Pricing and operational monitoring

FORMAT REQUIREMENTS:
- Use clear headings with # and ## for markdown
- Include specific numbers and percentages
- Use bullet points with hyphens (-)
- Do NOT use asterisks for bold text
- Use pound signs (£) and commas for numbers
- Be professional and data-driven
- Make actionable, insightful, and readable
"""
    try:
        client = get_client()
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt,
            config={"temperature": 0.4, "max_output_tokens": 8192},
        )
        return response.text if response.text else "Unable to generate report."
    except Exception as e:
        return f"⚠️ Report generation failed: {str(e)}"


# ---------------------------
# Example Usage
# ---------------------------
if __name__ == "__main__":
    try:
        kb_text = load_knowledge_base("inference/frostmart_knowledge_base.docx")
        print(chat_with_frostmart("What's the total estimated loss this year?", kb_text))
        print(chat_with_frostmart("Which product has the highest wastage?", kb_text))
        print(generate_frostmart_report(kb_text))
    except Exception as err:
        print(f"Error: {err}")
