import streamlit as st
import numpy as np

title = "Toxic Tweets"
st.title(title)

options = np.array(["BERT", "GPT"])
choice = st.selectbox("Select Model", options)

response = st.text_input("Enter Text")

if st.button("Process Text"):
    st.write("TODO Button Press. {}".format(response))