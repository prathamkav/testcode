import numpy as np
import pandas as pd
from ast import literal_eval
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
from .helpers import make_dataset, invert_multi_hot, make_model
from pymongo import MongoClient

client = MongoClient("mongodb+srv://pratham:8QpY4vbRD86bvkTh@staging.awytu.mongodb.net/?retryWrites=true&w=majority")




db = client.dashboard




test = db.tickets.find_one({"number":12})

print(test)



teams = db.accessList.find_one({"name": "Teams"})["teams"]
teams.remove('GridSec')
trainingList = []
filterDict = {
        "_id":1,
        "type": 1,
        "status":1,
        "body":1,
        "reason":1,
        "number": 1,
        "notes":1,
        "sites": 1,
        "requester": 1,
        "submitdate":  1,
        "subject": 1,
        "assign":1,
        "cip_impact": 1,
        'startDate': 1,
        'endDate':1,
        'name': 1,
        'email':1,
        "urgency":1,
        "category":1,
        "updated":1,
        'teams':1,
        "level": 1,
        "approval_date":1,
        "closed_date":1
    }
    # get data that will be used to train classifier
tickets = db.tickets.find({'teams':{"$exists":True}},filterDict)
for ticket in tickets:
        trainingList.append(ticket)
df = pd.DataFrame.from_dict(trainingList)
    # drop columns
df = df.drop(columns=["_id", "submitdate", "startDate", "endDate", "updated", "category", "approval_date", "closed_date"])
    

    # replace null or NaN values in teams with an empty list
df["teams"] = df["teams"].fillna("").apply(list)

    # replace null or NaN values in all columns with ""
df = df.fillna("")
    
    # combine reason, subject body and notes columns into one column to use for training
df['text-data'] = df[['reason', 'subject', 'body', 'notes']].agg(lambda x: ' '.join(x.values), axis=1).T
df["teams"] = df["teams"].astype(str)

df = df[~df["text-data"].duplicated()]
    # filtering for rare teams
df = df.groupby("teams").filter(lambda x: len(x) > 1)
    
    # convert strings to lists of strings for teams
df["teams"] = df["teams"].apply(
        lambda x: literal_eval(x)
    )
    # initial train and test split
test_split = 0.1
train_df, test_df = train_test_split(df, test_size=test_split, stratify=df["teams"].values)


    # splitting test set further into validation and new test sets
val_df = test_df.sample(frac=0.5)
test_df.drop(val_df.index, inplace=True)
    
print(f"Number of rows in training set: {len(train_df)}")
print(f"Number of rows in validation set: {len(val_df)}")
print(f"Number of rows in test set: {len(test_df)}")
        
print(train_df["text-data"].apply(lambda x: len(x.split(" "))).describe())
    
    
    ################# Multi-label Binarization ######################
    
    # preprocessing teams using StingLookup layer, used to binarize labels 
lookup = tf.keras.layers.StringLookup(output_mode="multi_hot")
lookup.adapt(teams)
vocab = lookup.get_vocabulary()
    #################################################################
    
    
    
    
    
    
    ##########Data preprocessing and tf.data.Dataset objects##########
    
    #creating initial datasets for testing and training
batch_size = 26
auto = tf.data.AUTOTUNE
train_dataset = make_dataset(train_df, lookup, batch_size, is_train=True)
validation_dataset = make_dataset(val_df, lookup, batch_size,is_train=False)
test_dataset = make_dataset(test_df, lookup, batch_size, is_train=False)
    ###################################################################
    
    
    
    
    ######################## Dataset Preview ###########################
text_batch, label_batch = next(iter(train_dataset))

for i, text in enumerate(text_batch[:10]):
        label = label_batch[i].numpy()[None, ...]
    ###################################################################
    
    
    
    
    ######################## Vectorization #############################
    #Calculate unique words present in reasons. Then create vectorization 
    # layer and map to the tf.data.Datasets created earlier
    
    # A batch of raw text will first go through the TextVectorization layer and it will generate their integer representations. 
    # Internally, the TextVectorization layer will first create bi-grams out of the sequences and then represent them using TF-IDF. 
    # The output representations will then be passed to the shallow model responsible for text classification.
vocabulary = set()
train_df["text-data"].str.lower().str.split().apply(vocabulary.update)
vocabulary_size = len(vocabulary)
text_vectorizer = layers.TextVectorization(
        max_tokens=vocabulary_size, ngrams=2, output_mode="tf_idf"
    )

    # `TextVectorization` layer needs to be adapted as per the vocabulary from our
    # training set.
with tf.device("/CPU:0"):
        text_vectorizer.adapt(train_dataset.map(lambda text, label: text))
train_dataset = train_dataset.map(
        lambda text, label: (text_vectorizer(text), label), num_parallel_calls=auto
    ).prefetch(auto)
validation_dataset = validation_dataset.map(
        lambda text, label: (text_vectorizer(text), label), num_parallel_calls=auto
    ).prefetch(auto)
test_dataset = test_dataset.map(
        lambda text, label: (text_vectorizer(text), label), num_parallel_calls=auto
    ).prefetch(auto)
    ######################################################################
    

    
    # Train model
epochs = 1

shallow_mlp_model = make_model(lookup.vocabulary_size())
    
shallow_mlp_model.compile(
        loss="binary_crossentropy", optimizer="adam", metrics=["categorical_accuracy"]
    )

history = shallow_mlp_model.fit(
        train_dataset, validation_data=validation_dataset, epochs=epochs
    )
_, categorical_acc = shallow_mlp_model.evaluate(test_dataset)
print(f"Categorical accuracy on the test set: {round(categorical_acc * 100, 2)}%.")
    
        


    # Create a model for inference.
model_for_inference = keras.Sequential([text_vectorizer, shallow_mlp_model])
    
    # Used to save model if model accuracy is above 90.56%
if round(categorical_acc * 100, 2)>90.56:
        model_for_inference.save(f"saved_model_{round(categorical_acc * 100)}/trained_model")


    # Create a small dataset just for demoing inference.
inference_dataset = make_dataset(test_df.sample(100), lookup, batch_size, is_train=False)
text_batch, label_batch = next(iter(inference_dataset))
predicted_probabilities = model_for_inference.predict(text_batch)

    # Perform inference.
for i, text in enumerate(text_batch[:10]):
        label = label_batch[i].numpy()[None, ...]
        # print(f"Reason: {text}")
        print(f"Label(s): {invert_multi_hot(vocab, label[0])}")
        top_label = [
            x
            for _, x in sorted(
                zip(predicted_probabilities[i], lookup.get_vocabulary()),
                key=lambda pair: pair[0],
                reverse=True,
            )
        ][:1]
        print(f"Predicted Label(s): ({', '.join([label for label in top_label])})")
        print(" ")
