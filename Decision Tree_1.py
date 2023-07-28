#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  1 14:38:51 2021

@author: yccg
"""

import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

df = pd.read_pickle('ModelOneHots')
Feature_set = df[['AvgSentLength', 'AvgChar', 'Avg TFIDF', 'vaderScore']]

# Accuracy/Percision/Recall/F1
def DT(a_type):
    # Train/Test
    F_train, F_test, T_train, T_test = train_test_split(Feature_set, a_type, test_size = 0.2, random_state = 1)
    # Scale
    cols = F_train.columns
    scaler = StandardScaler()
    F_train = scaler.fit_transform(F_train)
    F_test = scaler.transform(F_test)
    
    F_train = pd.DataFrame(F_train, columns=[cols])
    F_test = pd.DataFrame(F_test, columns=[cols])
    
    classifier = DecisionTreeClassifier()
    classifier = classifier.fit(F_train,T_train)
    
    T_pred = classifier.predict(F_test)
    accuracy = metrics.accuracy_score(T_test,T_pred)
    precision = metrics.precision_score(T_test,T_pred)
    recall = metrics.recall_score(T_test,T_pred)
    f1 = metrics.f1_score(T_test,T_pred)
    return accuracy,precision,recall,f1

d = defaultdict(list)

# for all type
a = t_list = [t for t in df['myerTypes']]
F_train, F_test, T_train, T_test = train_test_split(Feature_set, a, test_size = 0.2, random_state = 1)
cols = F_train.columns
scaler = StandardScaler()
F_train = scaler.fit_transform(F_train)
F_test = scaler.transform(F_test)

F_train = pd.DataFrame(F_train, columns=[cols])
F_test = pd.DataFrame(F_test, columns=[cols])

classifier = DecisionTreeClassifier()
classifier = classifier.fit(F_train,T_train)

T_pred = classifier.predict(F_test)
accuracy_a = metrics.accuracy_score(T_test,T_pred)
precision_a = metrics.precision_score(T_test,T_pred,average = 'weighted')
recall_a = metrics.recall_score(T_test,T_pred,average = 'weighted')
f1_a = metrics.f1_score(T_test,T_pred,average = 'weighted')


d = defaultdict(list)
type_list1 = ['Myer Type','I(1)/E','N(1)/S','P(1)/J','F(1)/T']

d['Accuracy'].append(accuracy_a)
d['Precision'].append(precision_a)
d['Recall'].append(recall_a)
d['F1'].append(f1_a)

for i in type_list1[1:]:
    t_list1 = [t for t in df[i]]
    result = DT(t_list1)
    d['Accuracy'].append(result[0])
    d['Precision'].append(result[1])
    d['Recall'].append(result[2])
    d['F1'].append(result[3])
    
data = dict(d)
df=pd.DataFrame(data,index=['Myer Type','I/E','N/S','P/J','F/N'])
print(df)










