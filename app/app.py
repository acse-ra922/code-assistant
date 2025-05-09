import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Set page title
st.set_page_config(page_title="Code Explainer Assistant", layout="centered")

# Title and instructions
st.title("🧠 Code Explainer Assistant")
st.write("Paste your Python code below and get an explanation of the complex parts.")

# Input box
code = st.text_area("✍️ Paste your Python code here:", height=300)

# Load model once when app starts
@st.cache_resource
def load_model():
    model_name = "Salesforce/codet5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Add a button to trigger explanation
if st.button("🔍 Generate Explanation"):
    if not code.strip():
        st.warning("Please paste some code first.")
    else:
        with st.spinner("Analyzing code..."):
            input_text = f"summarize: {code}"
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
            outputs = model.generate(**inputs, max_new_tokens=150)
            explanation = tokenizer.decode(outputs[0], skip_special_tokens=True)

        st.subheader("🧾 Explanation")
        st.write(explanation)
