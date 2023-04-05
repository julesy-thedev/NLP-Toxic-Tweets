import streamlit as st
import numpy as np
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

title = "Toxic Tweets"
st.title(title)

options = np.array(["BERT", "GPT"])
choice = st.selectbox("Select Model:", options)

response = st.text_input("Enter Text to Analyse:", "I am excited to begin working on this CS482 Project!")

if st.button("Process Text"):
    pred = classifier(response)
    st.write(pred)