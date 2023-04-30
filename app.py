import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd

#https://huggingface.co/course/chapter2/2?fw=pt
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

def process_button(logits, tweets):
    data = {
        "tweets": tweets,
        "label": [],
        "percent": []
    }

    probs = tf.math.sigmoid(logits)
    probs2label = np.zeros(probs.shape)

    probs2label[np.where(probs >= 0.5)] = 1

    for idx, row in enumerate(probs2label):
        last = np.flatnonzero(row)

        highest_level = highest_level_prob = None

        if len(last) > 0:
            highest = last[-1]

            highest_level = model.config.id2label[highest]
            highest_level_prob = probs[idx, highest]
        else:
            highest_level = "Non-Toxic"
            highest_level_prob = "N/A"

        data["label"].append(highest_level)
        data["percent"].append(highest_level_prob)

    return data

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

sample_tweets = ["Yo bitch Ja Rule is more succesful then you'll ever be whats up with you and hating you sad mofuckas...i should bitch slap ur pethedic white faces and get you to kiss my ass you guys sicken me. Ja rule is about pride in da music man. dont diss that shit on him. and nothin is wrong bein like tupac he was a brother too...fuckin white boys get things right next time.,", "Thank you for understanding. I think very highly of you and would not revert without discussion.", "Please stop. If you continue to vandalize Wikipedia, as you did to Homosexuality, you will be blocked from editing.", "I WILL BURN YOU TO HELL IF YOU REVOKE MY TALK PAGE ACCESS!!!!!!!!!!!!!"]

response = st.text_input("Enter Tweet to Analyse:", "I am excited to begin working on this Project!")

if st.button("Evaluate Tweet"):
    st.write(":blue[=== Results ===]")

    inputs = tokenizer(response, padding=True, return_tensors="tf")
    logits = model(**inputs).logits

    result = process_button(logits, response)

    df = pd.DataFrame(result)

    st.table(df)
    
if st.button("Load Sample Tweets"):
    st.write(":blue[=== Results ===]")

    inputs = tokenizer(sample_tweets, padding=True, return_tensors="tf")
    logits = model(**inputs).logits

    result = process_button(logits, sample_tweets)

    df = pd.DataFrame(result)

    st.table(df)