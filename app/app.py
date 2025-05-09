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
            try:
                # Preprocess code if it exceeds the model's max input length (512 tokens)
                input_text = f"Explain this Python code: {code}"
                inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512, padding=True)
                
                # Generate explanation
                outputs = model.generate(**inputs, max_new_tokens=300, num_beams=5, length_penalty=2.0)
                explanation = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Display the explanation
                st.subheader("🧾 Explanation")
                st.write(explanation)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
