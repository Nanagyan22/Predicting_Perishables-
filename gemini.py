# gemini.py — FrostMart UK Predictive Analytics Assistant

import os

try:
    # --- FIX 1: Using the modern, simplified SDK name as per your working example ---
    import google.genai as genai
except ModuleNotFoundError:
    genai = None

# Load Knowledge Base

def load_knowledge_base(path: str) -> str:
    """Load the FrostMart knowledge base from a Markdown file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Knowledge base file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# Gemini Client

def get_client():
    """Get or create Gemini client using the modern pattern."""
    if genai is None:
        # Note: We now check for the correct module name (google.genai)
        raise ModuleNotFoundError("google.genai module not installed. Please use 'pip install google-genai'.")
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    # --- FIX 2: Initialize the client using the correct modern class ---
    return genai.Client(api_key=api_key)



# AI CHAT FUNCTION — FrostMart UK

def chat_with_frostmart(user_question: str, knowledge_base: str, chat_history: list = None) -> str:
    """
    Chat with AI using the FrostMart UK knowledge base as context.
    """
    system_prompt = f"""You are a professional AI Assistant for FrostMart UK — a national retail chain specializing in fresh and perishable goods.

YOUR ROLE:
- Provide insights strictly from the FrostMart UK knowledge base and data
- Support responses with data-driven evidence (e.g., R², wastage %, revenue lift)
- Maintain a professional, factual tone suitable for business reports
- Use proper formatting for numbers (£, commas, %, decimals)
- If a question is outside FrostMart UK data, politely explain you can only answer within the FrostMart context

RESPONSE GUIDELINES:
1. Start with a clear and factual answer
2. Support with exact metrics or KPIs from the data
3. Use short bullet points for clarity
4. Keep answers under 10 sentences
5. Be analytical, concise, and data-focused
6. No markdown symbols (asterisks, italics, etc.)
7. Write in clean, readable English suitable for executives

KNOWLEDGE BASE:
{knowledge_base}
"""

    try:
        client = get_client() # client is now a genai.Client instance
        if chat_history is None:
            chat_history = []

        # Include last 10 messages + user question
        messages = [system_prompt] + [m["content"] for m in chat_history[-10:]] + [f"User Question: {user_question}"]
        full_prompt = "\n\n".join(messages)

        # --- FIX 3: Using client.models.generate_content with dictionary config ---
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt,
            config={
                "temperature": 0.3,
                "max_output_tokens": 1024,
            },
        )

        # Extract text directly from the response object
        return response.text if response.text else "No response generated."

    except Exception as e:
        # Note: If this fails, the error will be visible in the Streamlit app.
        return f"⚠️ Chat error: {str(e)}"



# COMPREHENSIVE BUSINESS REPORT GENERATOR — FrostMart UK

def generate_frostmart_report(knowledge_base: str) -> str:
    """
    Generate a detailed, professional FrostMart UK predictive analytics report.
    """
    prompt = f"""You are Francis Afful Gyan, a Business Intelligence Specialist for FrostMart UK.
Generate a comprehensive business analytics report using the knowledge base below.
Include insights on sales forecasting, wastage reduction, and operational optimization.

KNOWLEDGE BASE:
{knowledge_base}

Create a structured report with headings, numeric metrics, and clear recommendations.
"""

    try:
        client = get_client() # client is now a genai.Client instance
        
        # --- FIX 4: Using client.models.generate_content with dictionary config ---
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt,
            config={
                "temperature": 0.4,
                "max_output_tokens": 8192,
            },
        )

        # Extract text directly from the response object
        return response.text if response.text else "Unable to generate report."

    except Exception as e:
        return f"⚠️ Report generation failed: {str(e)}"



# EXAMPLE USAGE

if __name__ == "__main__":
    # Ensure GEMINI_API_KEY is set in environment for this block to run
    try:
        # Placeholder for actual knowledge base file path
        kb_text = load_knowledge_base("inference/frostmart_knowledge_base.md")
        print(chat_with_frostmart("Which category had the most wastage?", kb_text))
        print(generate_frostmart_report(kb_text))
    except Exception as err:
        print(f"Error: {err}")
