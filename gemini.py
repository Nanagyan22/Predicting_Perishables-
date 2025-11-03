# gemini.py — FrostMart UK Predictive Analytics Assistant

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
You are a professional AI Assistant for FrostMart UK — a national retail chain specializing in perishable goods.

YOUR ROLE:
- Provide insights strictly from the FrostMart UK knowledge base and data
- Support responses with data-driven evidence (e.g., R², wastage %, revenue lift)
- Maintain a professional, factual tone suitable for business reports
- Format numbers properly (£, commas, %, decimals)
- If a question is outside FrostMart UK data, politely say you can only answer within FrostMart context

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
    """Generate a detailed business analytics report."""
    if genai is None:
        return "⚠️ google.genai module not installed. Cannot generate report."

    prompt = f"""
You are Francis Afful Gyan, a Business Intelligence Specialist for FrostMart UK.
Generate a comprehensive business analytics report using the knowledge base below.
Include insights on sales forecasting, wastage reduction, and operational optimization.

KNOWLEDGE BASE:
{knowledge_base}

Create a structured report with headings, numeric metrics, and clear recommendations.
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


# Example Usage (run locally)
if __name__ == "__main__":
    try:
        kb_text = load_knowledge_base("inference/frostmart_knowledge_base.docx")
        print(chat_with_frostmart("Which category had the most wastage?", kb_text))
        print(generate_frostmart_report(kb_text))
    except Exception as err:
        print(f"Error: {err}")
