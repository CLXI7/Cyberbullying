import streamlit as st
import google.generativeai as genai

# -----------------------------
# APP CONFIGURATION
# -----------------------------
# Changed page title and icon
st.set_page_config(page_title="Cyberbullying Detection Tool", page_icon="🛡️", layout="centered")

st.title("🚨 AI Cyberbullying Detector")
# Updated description
st.write("Paste text below to check for potential cyberbullying, toxicity, or hate speech using a powerful AI model.")

# -----------------------------
# USER INPUT
# -----------------------------
# Removed name and career goal, added text input for content analysis
content_to_analyze = st.text_area(
    "Paste the comment or text to analyze:", 
    placeholder="e.g. You are terrible at this game and should quit."
)

# Optional instruction input (for fine-tuning the prompt)
context_instruction = st.text_input(
    "Optional: Provide specific context (e.g., 'This is from a gaming forum'):", 
    placeholder="e.g. A comment from a YouTube video."
)

# -----------------------------
# CONFIGURE GEMINI SDK
# -----------------------------
# NOTE: Replace with your actual API key
api_key = st.secrets["GEMINI_API_KEY"]

genai.configure(api_key=api_key)

# Initialize the Gemini model
gemini_model = genai.GenerativeModel('gemini-2.5-flash')

# -----------------------------
# FUNCTION TO GENERATE DETECTION RESULT
# -----------------------------
def analyze_content(prompt):
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred while analyzing content: {e}"

# -----------------------------
# SUBMIT BUTTON
# -----------------------------
if st.button("Analyze Text for Bullying"):
    if not content_to_analyze.strip():
        st.warning("Please paste some content to analyze first.")
    else:
        with st.spinner("Analyzing text for toxicity and cyberbullying..."):
            # CRITICAL: New, specialized prompt for detection
            prompt = (
                f"Analyze the following text for signs of cyberbullying, hate speech, or toxicity. "
                f"Based on a 1 (low risk) to 5 (high risk) scale, provide a **Risk Score**.\n"
                f"Then, provide a **detailed explanation** for your score, specifically identifying which parts, "
                f"if any, are problematic. Conclude with a clear **Classification** (e.g., 'Bullying Detected' or 'Not Bullying').\n\n"
                f"Text to Analyze: \"{content_to_analyze}\"\n"
                f"Context/Instructions: {context_instruction if context_instruction else 'None provided.'}"
            )
            result = analyze_content(prompt)
            
        st.subheader("Analysis Result:")
        st.info(result) # Using st.info for a structured result display

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("Cyberbullying detection powered by Streamlit and Google Gemini.")
