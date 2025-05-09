import streamlit as st
from transformers import pipeline

st.title("🧠 Code Explainer Assistant")
st.write("Paste your Python code below and get an explanation of complex sections.")

code = st.text_area("Your Python Code", height=300)

if code:
    with st.spinner("Analyzing code..."):
        prompt = (
            "You are an expert Python teacher. Explain the complex parts of the following code "
            "so a beginner can understand:\n\n"
            f"{code}\n\n"
            "Focus on logic, non-obvious steps, and advanced constructs."
        )

        generator = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1", device_map="auto")

        output = generator(prompt, max_new_tokens=512, do_sample=True, temperature=0.7)
        explanation = output[0]["generated_text"].split(prompt)[-1].strip()

        st.subheader("🧾 Explanation")
        st.write(explanation)
