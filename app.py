import streamlit as st
import numpy as np

#https://huggingface.co/course/chapter2/2?fw=pt
from transformers import pipeline, AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf

models = {
    "DistilBERT": "distilbert-base-uncased-finetuned-sst-2-english",
    "RoBERTa": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "XLM-RoBERTa": "cardiffnlp/xlm-roberta-base-sentiment-multilingual",
    "ELECTRA": "bhadresh-savani/electra-base-emotion"
}

title = "Toxic Tweets"
st.title(title)

options = np.array( list(models.keys()) )
choice = str(st.selectbox("Select Model:", options))

pt_model = models[choice]

tokenizer = AutoTokenizer.from_pretrained(pt_model)
model = TFAutoModelForSequenceClassification.from_pretrained(pt_model)

classifier = pipeline("text-classification", model=pt_model)

response = st.text_input("Enter Text to Analyse:", "I am excited to begin working on this CS482 Project!")

if st.button("Submit"):
    st.header(":blue[Results]")

    pred = classifier(response)
    st.write(pred)

    tokens = tokenizer(response, padding=True, truncation=True, return_tensors="tf")
    outputs = model(**tokens)
    predictions = tf.nn.softmax(outputs.logits, axis=-1)

    st.write(predictions)
    st.write(model.config.id2label)