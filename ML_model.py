import pickle
import uvicorn
import numpy as np
import nest_asyncio
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


#### Getting data from file
input_data = pd.read_csv('BankNote_Authentication.csv')

#### Independent and dependent feature
x = input_data.iloc[:, :-1]
y = input_data.iloc[:, -1]


#### Train test split
x_train, x_test,y_train,y_test=train_test_split(x,y, test_size=0.3,random_state=0)


#### Implement Random Forest Classifier
classifier = RandomForestClassifier()
classifier.fit(x_train,y_train)


#### Prediction
y_pred = classifier.predict(x_test)


#### Check accuracy
score = accuracy_score(y_test, y_pred)
score


#### Create pickle file using serialization
pickle_out = open("model_bin", "wb")
pickle.dump(classifier, pickle_out)
pickle_out.close()
classifier.predict([[2,5,2,23]])