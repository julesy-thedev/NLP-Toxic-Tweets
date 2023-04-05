import streamlit as st
import numpy as np

title = "Toxic Tweets"
st.title(title)

options = np.array(["BERT", "GPT"])
st.selectbox("Select Model", options)

st.text("Enter Text")

if st.button('Process Text'):
    st.write('TODO')
else:
    st.write('NOT TODO')
