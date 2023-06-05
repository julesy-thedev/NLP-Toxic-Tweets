---
title: Project Cesspool
emoji: 🤢
colorFrom: yellow
colorTo: orange
sdk: streamlit
app_file: app.py
pinned: true
---

# **PROJECT CESSPOOL**

Explore _Project Cesspool_ futher and see it in action [here](https://sites.google.com/njit.edu/project-cesspool/home)

## *Milestones*

Developing a Natureal Language Model to classify toxic 🤢 tweets using HugginFace, Streamlit, Tensorflow, Python and DistilBERT model.

> ### **Milestone 1**
To begin I created my development environment using Docker containers. The main purpose of the using Docker is to isolate the development environment from the host while providing the necessary packages and environment for Jupyter Notebooks.

> ### **Milestone 2**
For _milestone-2_, I was developed a streamlit app that allows the user to enter a text, select a pretrained model and get the sentiment analysis of the text using HuggingFace transformers library and HuggingFace Spaces.

Streamlite app is located [here](https://huggingface.co/spaces/julesy/toxic-tweets)

https://user-images.githubusercontent.com/45794969/230540060-2a790672-6e8c-4c14-8842-6237a88ff91d.mp4


> ### **Milestone 3**
For _milestone-3_, I was built a multi-headed model that’s capable of detecting different types of of toxicity like threats, obscenity, insults, and identity-based hate better than Perspective’s current models. The classifier was be developed using a pretrained language model of my choice, I chose DistilBERT.

Streamlite app is located [here](https://huggingface.co/spaces/julesy/toxic-tweets)

Model development and training can be found in the toxic_comments notebook [here](https://github.com/julesy-thedev/CS482-Toxic-Tweets/blob/milestone-3/toxic_comments.ipynb)

## *Results*

This model is a fine-tuned version of distilbert-base-uncased on an unknown dataset. It achieves the following results on the evaluation set:

Train Loss: 0.0256

Train Accuracy: 0.9935

Validation Loss: 0.0414

Validation Accuracy: 0.9922

Number of Epochs: 3

Training Size = 159557 tweets