from keras.models import load_model
new_model = load_model('ann.classifier')

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
file = input("Enter file name: ")
dataset2 = pd.read_csv(file)
test = dataset2.iloc[:, 3:13].values
yt = dataset2.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_X_2 = LabelEncoder()
test[:, 2] = labelencoder_X_2.fit_transform(test[:, 2])

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
test = np.array(ct.fit_transform(test), dtype=np.float)
test = test[:, 1:]

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
test = sc.fit_transform(test)
#test = sc.transform(test)

# Predicting the Test set results
pred = new_model.predict(test)
pred = (pred > 0.5)

surname = dataset2.iloc[:, 2].values
s = surname.tolist()
p = pred.tolist()

print("\n%-20s | %-20s " %("SURNAME","RESULT"))
print('-------------------------------------------------')

Dict = {} 
for key in s: 
    for value in p: 
        Dict[key] = value 
        p.remove(value) 
        break  
    
for key, value in Dict.items():
    #print(key, '\t\t:', value)
    print("%-20s | %-20s " %(key,value))

import csv
w = csv.writer(open("output.csv", "w"))
for key, val in Dict.items():
    w.writerow([key, val])
    
# Predicting a single new observation
"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""
'''new_prediction = new_model.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)
print(new_prediction)
'''

