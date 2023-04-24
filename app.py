import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd

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

sample_tweets = ["Yo bitch Ja Rule is more succesful then you'll ever be whats up with you and hating you sad mofuckas...i should bitch slap ur pethedic white faces and get you to kiss my ass you guys sicken me. Ja rule is about pride in da music man. dont diss that shit on him. and nothin is wrong bein like tupac he was a brother too...fuckin white boys get things right next time.,", "Thank you for understanding. I think very highly of you and would not revert without discussion.", "Please stop. If you continue to vandalize Wikipedia, as you did to Homosexuality, you will be blocked from editing.", "I WILL BURN YOU TO HELL IF YOU REVOKE MY TALK PAGE ACCESS!!!!!!!!!!!!!"]

data = {
  "tweets": sample_tweets,
  "label": [],
  "percent": []
}

if st.button("Load Tweets"):
    st.write(":blue[=== Results ===]")

    for i,j in enumerate(sample_tweets):
        tokens = tokenizer(j, return_tensors='tf')
        outputs = model(tokens)

        predictions = tf.nn.softmax(outputs.logits, axis=-1)

        predicted_amount = float(tf.math.reduce_max(predictions, axis=-1)[0])
        predicted_class_id = int(tf.math.argmax(predictions, axis=-1)[0])

        data["label"].append(model.config.id2label[predicted_class_id])
        data["percent"].append(predicted_amount)

    df = pd.DataFrame(data)

    st.table(df)