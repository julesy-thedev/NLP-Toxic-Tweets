import streamlit as st
import numpy as np

#https://huggingface.co/course/chapter2/2?fw=pt
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

#classifier = pipeline("sentiment-analysis")

title = "Toxic Tweets"
st.title(title)

options = np.array(["BERT", "GPT"])
choice = st.selectbox("Select Model:", options)

response = st.text_input("Enter Text to Analyse:", "I am excited to begin working on this CS482 Project!")

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

if st.button("Submit"):
    st.header(":blue[Results]")

    #pred = classifier(response)
    #st.write(pred)

    tokens = tokenizer(response, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**tokens)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    st.write(predictions)