---
title: Toxic Tweets
emoji: 🤢
colorFrom: yellow
colorTo: orange
sdk: streamlit
app_file: app.py
pinned: false
---

> # **Toxic Tweets**
Developing a Language Model to classify toxic 🤢 tweets using HugginFace, Streamlit and GitHub.

Jules Blount
31430956


> ### **Milestone 1**
To begin I should mention I already have a home server with docker already installed with multiple containers running.  
The tutorial I followed to install docker on my server is located [Here](https://docs.docker.com/engine/install/debian/)

*Docker runtime environment verification*:

![](images/docker_version.png)

*Python prompt from Python container*:

![](images/pytrhon_container.png)

> ### **Milestone 2**
For _milestone-2_, I was tasked to develop a streamlit app that allows the user to enter a text, select a pretrained model and get the sentiment analysis of the text using HuggingFace transformers library and HuggingFace Spaces.

Streamlite app is located [here](https://huggingface.co/spaces/julesy/toxic-tweets)

https://user-images.githubusercontent.com/45794969/230540060-2a790672-6e8c-4c14-8842-6237a88ff91d.mp4


> ### **Milestone 3**
For _milestone-3_, I was challenged to build a multi-headed model that’s capable of detecting different types of of toxicity like threats, obscenity, insults, and identity-based hate better than Perspective’s current models. The classifier was be developed using a pretrained language model of my choice, I chose DistilBERT.

Streamlite app is located [here](https://huggingface.co/spaces/julesy/toxic-tweets)

Model development and training can be found in the toxic_comments notebook [here](https://github.com/julesy-thedev/CS482-Toxic-Tweets/blob/milestone-3/toxic_comments.ipynb)
