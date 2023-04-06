import streamlit as st
import numpy as np

#https://huggingface.co/course/chapter2/2?fw=pt
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

models = {
    "DistilBERT": "distilbert-base-uncased-finetuned-sst-2-english"
}

classifier = pipeline("text-classification")

title = "Toxic Tweets"
st.title(title)

dict_keys = models.keys()
options = np.array(dict_keys)
choice = str(st.selectbox("Select Model:", options))

tokenizer = AutoTokenizer.from_pretrained(models[choice])
model = AutoModelForSequenceClassification.from_pretrained(models[choice])

response = st.text_input("Enter Text to Analyse:", "I am excited to begin working on this CS482 Project!")

if st.button("Submit"):
    st.header(":blue[Results]")

    pred = classifier(response)
    st.write(pred)

    tokens = tokenizer(response, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**tokens)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    st.write(predictions)
    st.write(model.config.id2label)