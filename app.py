import streamlit as st
import numpy as np

#https://huggingface.co/course/chapter2/2?fw=pt
from transformers import pipeline, AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf

models = {
    "DistilBERT": "distilbert-base-uncased-finetuned-sst-2-english",
    "RoBERTa": "roberta-large-mnli",
    "XLM-RoBERTa": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    "ELECTRA": "bhadresh-savani/electra-base-emotion"
}

title = "Toxic Tweets"
st.title(title)

options = np.array( list(models.keys()) )
choice = str(st.selectbox("Select Base Model:", options))
pre_model = models[choice]

st.write("Model Used: ", pre_model)

tokenizer = AutoTokenizer.from_pretrained(pre_model)
model = TFAutoModelForSequenceClassification.from_pretrained(pre_model)

classifier = pipeline("text-classification", model=pre_model)

response = st.text_input("Enter Text to Analyse:", "I am excited to begin working on this CS482 Project!")

if st.button("Submit"):
    st.write(":blue[=== Results ===]")

    tokens = tokenizer(response, return_tensors='tf')
    outputs = model(tokens)

    predictions = tf.nn.softmax(outputs.logits, axis=-1)

    predicted_amount = int(tf.math.reduce_max(outputs.logits, axis=-1)[0])
    predicted_class_id = int(tf.math.argmax(outputs.logits, axis=-1)[0])

    pred = classifier(response)
    st.write(pred)

    st.write(predictions)

    st.write("This sentence can be interpreted as: ", predicted_amount, model.config.id2label[predicted_class_id])