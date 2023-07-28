#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  9 23:19:19 2021

@author: yccg
"""

import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.preprocessing import StandardScaler

df = pd.read_pickle('ModelOneHots')
Feature_set = df[['AvgSentLength', 'AvgChar', 'Avg TFIDF', 'vaderScore']]

# Confusion Matrix
def CM(type_one):
    # Train/Test
    F_train, F_test, T_train, T_test = train_test_split(Feature_set, type_one, test_size = 0.2, random_state = 1)
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
    CM = confusion_matrix(T_test,T_pred)
    return CM

type_list2 = ['I(1)/E','N(1)/S','P(1)/J','F(1)/T']
for i in type_list2:
    t_list2 = [t for t in df[i]]
    result = CM(t_list2)
    confusion = result 
    classes = list(set(t_list2)) # set class is 0 and 1
    classes.sort()
    
    plt.imshow(confusion, cmap=plt.cm.Blues) 
    indices = range(len(confusion))
    plt.xticks(indices, classes)
    plt.yticks(indices, classes)
    plt.colorbar()
    
    plt.xlabel('Predict')
    plt.ylabel('Fact')
    for first_index in range(len(confusion)):
        for second_index in range(len(confusion[first_index])):
            plt.text(first_index, second_index, confusion[first_index][second_index])
    plt.show()


# Decision Tree
list_column = ['AvgSentLength', 'AvgChar', 'Avg TFIDF', 'vaderScore','I(1)/E','N(1)/S','P(1)/J','F(1)/T']
data1 = df[list_column]

for i in list_column[4:]:
    t_list3 = [t for t in df[i]]
    F_train, F_test, T_train, T_test = train_test_split(Feature_set, t_list3, test_size = 0.2, random_state = 1)

    
    classifier = DecisionTreeClassifier(max_depth = 4)
    clf = classifier.fit(F_train,T_train)
    clf = clf.fit(F_test, T_test)
    
    plt.figure(figsize=(25,10))
    a = plot_tree(clf,
                  feature_names=['AvgSentLength', 'AvgChar', 'Avg TFIDF', 'vaderScore'],  
                  class_names=['0','1'],
                  filled=True, 
                  rounded=True, 
                  fontsize=14)
