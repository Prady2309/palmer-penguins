import pandas as pd
from sklearn.ensemble import RandomForestClassifier
# import streamlit as st

penguins = pd.read_csv("penguins_cleaned.csv")

df = penguins.copy()

target = 'species'
encode = ['sex', 'island']

for col in encode :
    dummy = pd.get_dummies(df[col], prefix=col)   # 0s and 1s
    df = pd.concat([df,dummy], axis=1)
    del df[col]

target_mapper = {'Adelie':0, 'Chinstrap':1, 'Gentoo':2}

def target_encode(val) :
    return target_mapper[val]

# individually send the species to the function target_encode
df['species'] = df['species'].apply(target_encode)  

#seperating x and y
x = df.drop('species', axis=1)
y = df['species']

# Build a random forest model
classifier = RandomForestClassifier()
classifier.fit(x, y)

#Saving the model
import pickle
pickle.dump(classifier, open('penguins_clf.pkl', 'wb'))
# put all the contents of the classifier into the file
