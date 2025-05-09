import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Local Code Explainer")

st.title("Local Code Explainer (Free & Fast)")

code = st.text_area("Paste your code here:", height=300)
task = st.selectbox("Select Task", ["Explain Code"])  # You can add more later

@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="google/flan-t5-base")

model = load_model()

if st.button("Run"):
    if not code.strip():
        st.warning("Please enter some code.")
    else:
        prompt = f"Explain the following Python code in simple terms:\n\n{code}"
        with st.spinner("Generating explanation..."):
            response = model(prompt, max_length=512, do_sample=False)
        st.subheader("Explanation:")
        st.write(response[0]['generated_text'])
