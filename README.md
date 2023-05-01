---
title: Toxic Tweets
emoji: 🤢
colorFrom: yellow
colorTo: orange
sdk: streamlit
app_file: app.py
pinned: false
---

# Project-Cesspool

## Description

The internet has been a tool that has offered unprecedented utility to mankind, but it has also become a tool that can foster hatred and even cause individuals to commit suicide. The threat of abuse and harassment online means that many people stop expressing themselves and give up on seeking differing opinions. Platforms struggle to effectively facilitate conversations, leading many communities to limit or completely shut down user comments. This has led to a decrease in the diversity of opinions and ideas being shared online.

It is important for platforms to take steps towards creating a safer online environment for users. The primary focus for Project Cesspool is to develop a tool which can allow platform owners to use a multi-headed model that’s capable of detecting different types of toxicity like threats, obscenity, insults, and identity-based hate.

## Table of Contents

- [How Project Cesspool Was Trained](#training)
- [Milestones](#milestones)

---

## Training

We began by importing the required packes needed. A majority of tools used were provided by Huggingface and Tensorflow. We also initialized global variables for training batch size and number of epochs, these will be used throughout training and provide a single source of truth should training parameters need to be changed. 

    import os
    import pandas as pd
    import numpy as np
    import tensorflow as tf

    from google.colab import drive
    from zipfile import ZipFile
    from datasets import load_dataset
    from huggingface_hub import notebook_login
    from tensorflow.keras.losses import BinaryCrossentropy
    from tensorflow.keras.optimizers.schedules import PolynomialDecay
    from tensorflow.keras.optimizers import Adam
    from transformers import TFAutoModelForSequenceClassification, TFAutoModelForSequenceClassification, DataCollatorWithPadding, AutoTokenizer

    batch_size = 8
    num_epochs = 3

Since we are using Google Colab to perform the training due to GPU availability, we need to mount our Google Drive as local store on Colab's VM. A unzip function was also created to unzip the training dataset once recieved from Kaggle.

    drive.mount('/content/gdrive', force_remount=True)
    def unzip(zip_file_loc):

    with ZipFile(zip_file_loc, 'r') as zipFile:
        zipFile.extractall()
    zipFile.close()

Next we set the KAGGLE_CONFIG_DIR to the correct directory in our Google Drive to locate kaggle.json which contains our API key for Kaggle. Once the Kaggle API is functional we download the required dataset and unzip the necessary files and remove irrelevant zip files. With our data downloaded into our local directory, main preprocessing and training can begin.

    os.environ['KAGGLE_CONFIG_DIR'] = "/content/gdrive/My Drive/Colab Notebooks/"
    path = "/content/gdrive/My Drive/Colab Notebooks/Project/"
    dataset_name = "jigsaw-toxic-comment-classification-challenge"

    %cd "/content/gdrive/My Drive/Colab Notebooks/Project/"

    !kaggle competitions download -c $dataset_name

    unzip(path + dataset_name + ".zip")
    unzip("test.csv.zip")
    unzip("test_labels.csv.zip")
    unzip("train.csv.zip")

    !rm *.zip

With the CSV's on our local machine our datasets can be loaded from local files using the load_dataset() function. The result is a DatasetDict object which contains the training set. The training set contains several columns (id, comment_text, toxic, severe_toxic etc) and a variable number of rows, which are the number of elements in the training data(159572 for this dataset)

    raw_train_dataset = load_dataset("csv", data_files="train.csv")

Since the dataset does not comes with a test dataset, we have to create one using the train_test_split() method. Here we are splitting the initial train dataset into 2. We will take 20% of our raw dataset and use for testing and the remainding 80% we will use for training. We will then assign these two dataset to 2 new variables for later use. 

    raw_train = raw_train_dataset["train"]

    test = raw_train.train_test_split(test_size=0.2, seed=42)

    raw_train_dataset["train"] = test["train"]
    raw_train_dataset["test"] = test["test"]

Before you start training your model, we neeed to create a map of the expected ids to their labels with id2label and label2id. We accomplish this by using python's list comprehension using the features in the raw dataset and extracting their values using the keys() method. What we are left with is a list of labels obtained from the column names in the CSV. Finally we enumerate over these labels and assign ids to labels and labels to ids.

    labels = [label for label in raw_train_dataset["train"].features.keys() if label not in ["id", "comment_text"]]

    id2label = {idx:label for idx, label in enumerate(labels)}
    label2id = {label:idx for idx, label in enumerate(labels)}

To preprocess the dataset, we need to convert the text to numbers the model can make sense of using a Tokenizer (distilbert-base-uncased). The Tokenizer performs 2 actions: tokenization, followed by the conversion to input IDs. Tokenization splits the text into words and and conversion maps the words to numbers for representation. Next we define a function which will be using for mapping. The Dataset.map() method method works by applying the tokenize_function on each element of the dataset; This function takes a dictionary and returns a new dictionary with the keys input_ids, attention_mask, and token_type_ids added. tokenize_function also converts the colum values into a single list as the label. It first creates a 2d numpy and fill it with zeros. Then we grab list of values from each colum and save it to a dictionary. We finally go through the labels and update the values in each colum to the obtained values. This numpy array is then converted to a list and assigned to the "labels" column of the dataset.

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize_function(batched_dataset):
        tokenized_batch_elements = tokenizer(batched_dataset["comment_text"], truncation=True)

        num_labels = len(labels)

        labels_2d = np.zeros((len(batched_dataset["comment_text"]), num_labels))

        colums_2_labels = {x: batched_dataset[x] for x in batched_dataset.keys() if x in labels}

        for i, label in enumerate(labels):
            labels_2d[:, i] = colums_2_labels[label]

        tokenized_batch_elements["labels"] = labels_2d.tolist()

        return tokenized_batch_elements

    tokenized_dataset = raw_train_dataset.map(tokenize_function, remove_columns=raw_train_dataset["train"].column_names, batched=True)

Next we defined a collate function that will apply the correct amount of padding to the items of the dataset we batched together. Next we use the to_tf_dataset() method to help load batches and collate them and converted our dataset into format that’s ready for training.

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

    tf_train_dataset = tokenized_dataset["train"].to_tf_dataset(
    columns = ["attention_mask", "input_ids"],
    label_cols = ["labels"],
    shuffle = True,
    collate_fn = data_collator,
    batch_size = batch_size,
    )

    tf_validation_dataset = tokenized_dataset["test"].to_tf_dataset(
    columns = ["attention_mask", "input_ids"],
    label_cols = ["labels"],
    shuffle = False,
    collate_fn = data_collator,
    batch_size = batch_size,
    )

We ensured high model performance by improving  loss decline using a slowly reducing learning rate over the course of training. We implemented a PolynomialDecay which simply decayed the learning rate from an initial value to the final value over the course of training. Finally our customer learning rate as assigned to our model's optimizer.

    num_train_steps = len(tf_train_dataset) * num_epochs

    lr_scheduler = PolynomialDecay(initial_learning_rate = 5e-5, 
                                end_learning_rate = 0.0, 
                                decay_steps = num_train_steps)

    opt = Adam(learning_rate = lr_scheduler)

Finally we initialized our model using the from_pretrained() method. Here we used the distilbert-base-uncased as our pretrained model, removed it's head and attached a new head capable of multi_label_classification with 6 classes. multi_label_classification means that the input can be in some, all or none of the classes. Finally, once the model has been initialized it must be complied. In this case the distilbert-base-uncased model as assigned a custom optimizer, the loss function was get to BinaryCrossentropy since we dealing with multi_label_classification and probabilties and the metrics used when calculating the loss was based on accuracy since this was a classification task.

    model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", problem_type="multi_label_classification", num_labels=len(labels), id2label=id2label, label2id=label2id)

    model.compile(optimizer = opt,
                loss = BinaryCrossentropy(from_logits=True),
                metrics = ["accuracy"]
                )

Finally the model was passed the training and test datasets and asked to fit the model to the data over 3 epochs.

    model.fit(tf_train_dataset, validation_data = tf_validation_dataset, epochs = num_epochs)

Once the model was finish training, its final state was uploaded to HuggingFace so that everyone can use the model. In the end the model attained a Validation Accuracy: 0.9922 over 3 epochs.

    notebook_login()
    model.push_to_hub("toxic-comments-distilbert")
    tokenizer.push_to_hub("toxic-comments-distilbert")

## Milestones

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

> ### **Milestone 4**
Project Cesspool's Landing page can be found at [Project_Cesspool_Landing](https://sites.google.com/njit.edu/project-cesspool/home) 

Landing page also contains the demo video. A direct link to the vide can be found at [Demo](https://youtu.be/Al55zTl5AlI)

Documentation can be found at the [Top](#training)