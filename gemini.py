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
def generate_frostmart_report(knowledge_base: str = "") -> str:
    """
    Generates a full FrostMart UK Business Insights Report using Gemini.
    Accepts an optional knowledge_base string (DOCX contents) to ground the report.
    """
    if genai is None:
        return "⚠️ google.genai module not installed. Cannot generate report."

    # Safety: ensure knowledge base is present (not blocking, but noted to the model)
    kb_note = knowledge_base if knowledge_base else "Knowledge base content not provided; use available dataset and defaults."

    report_prompt = f"""
You are Francis Afful Gyan, a Business Intelligence Specialist at FrostMart UK.
Generate a comprehensive, professional, and well-formatted business insights report for FrostMart UK,
modeled after the example provided (Iron Core Fitness). The report must be long, clearly structured,
and suitable for executive leadership.

Requirements:
- Use plain text only (no markdown headings like #, no bold/asterisk symbols, no emojis).
- Produce exactly nine numbered sections with descriptive titles:
  1. Executive Summary
  2. Financial Performance Analysis
  3. Product Category Performance
  4. Regional Sales and Wastage Trends
  5. Demand Forecasting Model Evaluation
  6. Supply Chain and Inventory Insights
  7. Key Challenges and Opportunities
  8. Strategic Recommendations
  9. Conclusion
- Write full paragraphs (not bullet-only), include clear numeric KPIs and examples.
- Use realistic FrostMart values where applicable:
  - Annual revenue around £255,716,700 (or state estimated revenue using available data)
  - Wastage rate ~ 7.8%
  - Estimated annual loss £12,200,000
  - Predicted waste reduction savings 30–40%
  - Model performance: R² ~ 0.9959, MAPE ~ 0.86%
  - Expected revenue uplift 10–20%
- For currency, always use pounds (£) with commas and two decimal places (e.g., £12,200,000.00).
- Provide actionable, quantified recommendations. Each recommendation should include a "Success Metric".
- Minimum length: aim for 1,200–1,800 words.
- Start the output with the following lines (plain text exactly as shown):
FROSTMART UK BUSINESS INSIGHTS REPORT
Date: November 2025
Prepared by: Francis Afful Gyan, Business Intelligence Specialist

Knowledge base (for reference):
{kb_note}
"""

    try:
        client = get_client()
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=report_prompt,
            config={"temperature": 0.35, "max_output_tokens": 16000},
        )

        # Safely extract text
        raw = ""
        if hasattr(response, "text"):
            raw = response.text or ""
        elif isinstance(response, str):
            raw = response
        else:
            raw = str(response)

        # Clean up stray markdown/symbols that the model might include despite instructions
        clean_text = (
            raw.replace("#", "")
               .replace("*", "")
               .replace("**", "")
               .replace("`", "")
               .strip()
        )

        # Ensure report begins with required title block; if not, prepend it
        header = (
            "FROSTMART UK BUSINESS INSIGHTS REPORT\n"
            "Date: November 2025\n"
            "Prepared by: Francis Afful Gyan, Business Intelligence Specialist\n\n"
        )
        if not clean_text.startswith("FROSTMART UK BUSINESS INSIGHTS REPORT"):
            clean_text = header + clean_text

        return clean_text

    except Exception as e:
        return f"⚠️ Report generation failed: {str(e)}"

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
