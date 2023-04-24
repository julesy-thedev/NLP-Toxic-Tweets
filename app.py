import streamlit as st
import numpy as np
import tensorflow as tf

#https://huggingface.co/course/chapter2/2?fw=pt
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

models = {
    "toxic-comments-distilbert": "julesy/toxic-comments-distilbert"
}

title = "Toxic Tweets"
st.title(title)

options = np.array( list(models.keys()) )
choice = str(st.selectbox("Custom Finetuned Model:", options))
pre_model = models[choice]

st.write("Fine Tuned DistilBert model for identifying Toxic Comments.")

tokenizer = AutoTokenizer.from_pretrained(pre_model)
model = TFAutoModelForSequenceClassification.from_pretrained(pre_model)

sample_tweets = ["I hate you!", "you're a dumbass", "thats pretty bad", "damn, i wish i didnt do that"]

if st.button("Load Tweets"):
    st.write(":blue[=== Results ===]")

    col1, col2, col3 = st.columns(3)

    for i in sample_tweets:
        tokens = tokenizer(i, return_tensors='tf')
        outputs = model(tokens)

        predictions = tf.nn.softmax(outputs.logits, axis=-1)

        predicted_amount = float(tf.math.reduce_max(predictions, axis=-1)[0])
        predicted_class_id = int(tf.math.argmax(predictions, axis=-1)[0])

        col1.write(i)
        col2.write(model.config.id2label[predicted_class_id])
        col3.write(predicted_amount)