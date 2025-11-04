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



# Load Knowledge Base

def load_knowledge_base(path: str) -> str:
    """Load the FrostMart knowledge base from a DOCX file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Knowledge base file not found: {path}")
    if docx is None:
        raise ModuleNotFoundError("python-docx module not installed. Install via 'pip install python-docx'.")
    
    document = docx.Document(path)
    full_text = "\n".join([para.text for para in document.paragraphs])
    return full_text


# Gemini Client

def get_client():
    """Create Gemini AI client."""
    if genai is None:
        raise ModuleNotFoundError(
            "google.genai module not installed. Install via 'pip install google-genai'."
        )
    return genai.Client(api_key=GEMINI_API_KEY)



# Chat with FrostMart AI

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



# Generate FrostMart Analytics Report
def generate_frostmart_report(knowledge_base: str) -> str:
    """Generate a comprehensive FrostMart UK business analytics report."""
    if genai is None:
        return "⚠️ google.genai module not installed. Cannot generate report."

    prompt = f"""
You are Francis Afful Gyan, a Business Intelligence Specialist for FrostMart UK.

Your task:
Generate a complete, detailed, and data-driven **Business Insights Report** for FrostMart UK, in the style and depth of a corporate business intelligence report (like the Iron Core Fitness example). The report must be **long, sectioned, and insight-rich**, not summarized.

Use the knowledge base provided below as your reference.
Include numeric details, KPIs, comparisons, and clear recommendations.

KNOWLEDGE BASE:
{knowledge_base}

FORMAT AND STRUCTURE REQUIREMENTS:
The output must strictly follow this structure and produce a comprehensive report with 8–10 sections:

# FROSTMART UK BUSINESS INSIGHTS REPORT

Date: November 2025  
Prepared by: Francis Afful Gyan, Business Intelligence Specialist  

1. Executive Summary
Provide a complete overview of FrostMart UK's performance in sales, wastage, profitability, and model-driven decision-making. Include specific figures like revenue uplift, wastage reduction, and AI accuracy metrics. Keep it at least 3 paragraphs long.

2. Financial Performance Analysis
- Include total revenue, total wastage cost, estimated losses (£12.2M), potential uplift (10–20%), and profit margin analysis.
- Discuss trends, sustainability impact, and cash flow implications.

3. Product and Regional Insights
- Identify top-selling and highest-wastage product categories.
- Compare regional performance (London, Midlands, South West, etc.)
- Mention specific improvements or risks.

4. Sales, Pricing, and Marketing Insights
- Analyze promotional impacts, peak sales months, and seasonal sales trends.
- Discuss marketing ROI and discount strategy effectiveness.

5. Operational and Supply Chain Performance
- Discuss inventory accuracy, overstocking, and procurement efficiency.
- Highlight waste reduction targets and logistics optimization.

6. Predictive Model Development and Performance
- Explain which models were used (Linear Regression, Random Forest, Gradient Boosting).
- Include performance metrics: R², RMSE, MAE, and MAPE.
- Describe deployment via Streamlit and integration with FrostMart’s procurement systems.

7. AI Deployment & Integration Strategy
- Detail how AI predictions are integrated into business workflows.
- Explain weekly retraining, data sources, and batch forecasting.
- Include mention of the Streamlit app modules (single prediction, batch CSV, chat assistant).

8. Key Challenges and Opportunities
List at least 4 challenges and 4 opportunities, focusing on wastage, demand forecasting, data quality, and regional sales variation.

9. Strategic Recommendations
Provide actionable, quantifiable business recommendations with success metrics. Each should include a “Success Metric” like:
- “Reduce wastage by 15% within 6 months.”
- “Improve regional profitability in London by 10%.”

10. Conclusion
Summarize FrostMart UK’s business health, projected gains from predictive analytics, and next steps for continuous improvement.

STYLE RULES:
- Write in professional corporate tone, no asterisks or emojis.
- Use markdown-style headings (#, ##, etc.).
- Include all currency in pounds (£) and formatted with commas.
- Write at least 1,200–1,800 words minimum.
- Never summarize too early — fully elaborate insights.
"""
    try:
        client = get_client()
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt,
            config={"temperature": 0.45, "max_output_tokens": 16000},
        )
        return response.text if response.text else "Unable to generate report."
    except Exception as e:
        return f"⚠️ Report generation failed: {str(e)}"


# Example Usage

if __name__ == "__main__":
    try:
        kb_text = load_knowledge_base("inference/frostmart_knowledge_base.docx")
        print(chat_with_frostmart("What's the total estimated loss this year?", kb_text))
        print(chat_with_frostmart("Which product has the highest wastage?", kb_text))
        print(generate_frostmart_report(kb_text))
    except Exception as err:
        print(f"Error: {err}")
