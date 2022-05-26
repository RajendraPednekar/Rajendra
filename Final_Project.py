# -*- coding: utf-8 -*-
"""
Created on Fri May 20 13:13:44 2022

@author: sathi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


sample = pd.read_excel("C:/Users/Rajendra/OneDrive/Desktop/Rajendra Pednekar/sample_data.xlsx")
sample.info()
sample.head()

sample.describe()

#### DPP
##Handling duplicates
sample.duplicated().sum()

##Missing values
sample.isna().sum()

##Zero variance
sample.var()
#Drop zero variance column
sample = sample.drop("Mode_Of_Transport", axis = 1)
sample = sample.drop("Test_Booking_Date", axis = 1)
sample = sample.drop("Sample_Collection_Date", axis = 1)
sample = sample.drop("Patient_ID", axis = 1)
sample = sample.drop("Agent_ID", axis = 1)
sample.info()

#Label encoding for categorical features
cat_col = ["Patient_Gender", "Test_Name", "Sample", "Way_Of_Storage_Of_Sample", "Cut-off Schedule", "Traffic_Conditions"]

lab = LabelEncoder()
mapping_dict ={}
for col in cat_col:
    sample[col] = lab.fit_transform(sample[col])
 
    le_name_mapping = dict(zip(lab.classes_,
                        lab.transform(lab.classes_))) #To find the mapping while encoding
 
    mapping_dict[col]= le_name_mapping
print(mapping_dict)

#Model Building
rf = RandomForestClassifier(n_estimators=5000, n_jobs=3, random_state=42, max_depth = 3)
rf.fit(sample.iloc[:,:15], sample.Reached_On_Time)

# saving the model
# importing pickle
import pickle
pickle.dump(rf, open('project.pkl', 'wb'))

# load the model from
project = pickle.load(open('project.pkl', 'rb'))

# checking for the results
list_value = pd.DataFrame(sample.iloc[:,:15])
list_value

print(project.predict(list_value))
