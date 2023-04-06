import streamlit as st
import numpy as np

#https://huggingface.co/course/chapter2/2?fw=pt
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
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

response = st.text_input("Enter Text to Analyse:", "I am excited to begin working on this CS482 Project!")

if st.button("Submit"):
    st.write(":blue[=== Results ===]")

    tokens = tokenizer(response, return_tensors='tf')
    outputs = model(tokens)

    predictions = tf.nn.softmax(outputs.logits, axis=-1)

    predicted_amount = float(tf.math.reduce_max(predictions, axis=-1)[0])
    predicted_class_id = int(tf.math.argmax(predictions, axis=-1)[0])

    st.markdown("This sentence can be interpreted as: _:green[{:.2%}] {}_".format(predicted_amount, model.config.id2label[predicted_class_id]))