# %%
import pandas as pd
import pickle
import numpy as np
import os.path

from tensorflow import keras
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pymongo import MongoClient
from helpers import make_dataset, invert_multi_hot, make_model,preprocess, prediction
from string import digits
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier




from pymongo import MongoClient

client = MongoClient("mongodb+srv://prajwal:wSnhpJZel3jMaesA@staging.awytu.mongodb.net/?retryWrites=true&w=majority")

db = client.dashboard




teams = db.accessList.find_one({"name": "Teams"})["teams"]
teams.remove('GridSec')

filterDict = {
        
        "type": 1,
        "status":1,
        "body":1,
        "reason":1,
        "number": 1,
        "notes":1,
        "sites": 1,
        "requester": 1,
        
        "subject": 1,
        "assign":1,
        "cip_impact": 1,
        
        
        'name': 1,
        'email':1,
        "urgency":1,
        "category":1,
        "severity":1,
        'teams':1,
        "level": 1,
        
    
    }
tickets = db.tickets.find({'teams':{"$exists":True}},filterDict)


df = pd.DataFrame.from_dict(list(tickets))
print(df)

df["teams"] = df["teams"].fillna("").apply(list)

df = df.fillna("")

df['text_data'] = df[['reason', 'subject', 'body', 'notes']].agg(lambda x: ' '.join(x.values), axis=1)


df["teams"] = df["teams"].astype(str)

df["severity"] = df["severity"].fillna("")

df["severity"] = df["severity"].astype(str)


# Preprocess data
df['processed_body'] = df['text_data'].apply(preprocess)

# Split data into training and testing sets
train_size = int(len(df) * 0.8)
train_data = df[:train_size]
test_data = df[train_size:]

# Vectorize text using bag-of-words approach
vectorizer = CountVectorizer(stop_words='english')
train_features = vectorizer.fit_transform(train_data['processed_body'])
test_features = vectorizer.transform(test_data['processed_body'])


x_axis = vectorizer.fit_transform(df['processed_body'])
x_train, x_test, y_train, y_test = train_test_split(x_axis, df['severity'], test_size=0.4, random_state=0)


rf = RandomForestClassifier(n_estimators=150, random_state=42)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)

# #params = {'objective': 'multi:softmax', 'num_class': len(df['severity'].unique())}
# dtrain = xgb.DMatrix(x_train, y_train)
# dtest = xgb.DMatrix(x_test, y_test)
# xgboost_model = xgb.train(params, dtrain, num_boost_round=100)

#model for severity 
#severity_model = MultinomialNB()
#severity_model.fit(x_train, y_train)

#make prediction for testing set 
#y_pred = severity_model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("Severity Prediction Accuracy: %.2f%%" % (accuracy * 100.0))

#print('Severity Prediction Accuracy:', accuracy_score(y_test, y_pred))

teams_x_train, teams_x_test, teams_y_train, teams_y_test = train_test_split(x_axis, df['teams'], test_size=0.4, random_state=0)


# Train teams using random forest classifier
rf_teams = RandomForestClassifier(n_estimators=150, random_state=42)
rf_teams.fit(teams_x_train, teams_y_train)
rf_teams_y_pred = rf_teams.predict(teams_x_test)

# Evaluate classifier on test set
accuracy_teams = accuracy_score(teams_y_test, rf_teams_y_pred)
print("Teams Prediction Accuracy: %.2f%%" % (accuracy_teams * 100.0))


#test ticket description

description = "Site wide inverter warranty repairs"
print(description)

#predicting severity
# predicted_severity_team= predict_team(teams_model,vectorizer,description)

predicted_severity_team = prediction(rf_teams, rf, vectorizer, description)
print("The team and severity for the above description is,", predicted_severity_team)



# if accuracy > 0.8:
#     model_name = 'ticket_classifier.pkl'
#     # Save the model
#     with open(model_name, 'wb') as f:
#         pickle.dump(clf, f)
#     # Save the vectorizer
#     with open('vectorizer.pkl', 'wb') as f:
#         pickle.dump(vectorizer, f)

# model_name = 'ticket_classifier.pkl'
# if os.path.isfile(model_name):
#     print("Model file exists")
# else:
#     print("Model file does not exist")


