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
choice = str(st.selectbox("Finetuned Model:", options))
pre_model = models[choice]

st.write("Fine Tuned DistilBert model for identifying Toxic Comments.")

tokenizer = AutoTokenizer.from_pretrained(pre_model)
model = TFAutoModelForSequenceClassification.from_pretrained(pre_model)

response = st.text_input("Enter Text to Analyse:", "I am excited to begin working on this Project!")
sample_tweets = ["I hate you!", "you're a dumbass", "thats pretty bad", "damn, i wish i didnt do that"]

if st.button("Submit"):
    st.write(":blue[=== Results ===]")

    col1, col2, col3 = st.columns(3)

    for i in sample_tweets:
        tokens = tokenizer(response, return_tensors='tf')
        outputs = model(tokens)

        predictions = tf.nn.softmax(outputs.logits, axis=-1)

        predicted_amount = float(tf.math.reduce_max(predictions, axis=-1)[0])
        predicted_class_id = int(tf.math.argmax(predictions, axis=-1)[0])

        col1.write(i)
        col2.write(model.config.id2label[predicted_class_id])
        col3.write(predicted_amount)

        #st.markdown("This sentence can be interpreted as: _:green[{:.2%}] {}_".format(predicted_amount, model.config.id2label[predicted_class_id]))