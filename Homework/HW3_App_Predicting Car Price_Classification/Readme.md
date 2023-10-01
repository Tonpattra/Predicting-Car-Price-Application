# HW3: Predicting Car Price Classification
The dataset is the same homeworks 1 but Implementation Based on 03 - Regularization.ipynb and modify LinearRegression()
![image](https://github.com/Tonpattra/Machine-Learning/assets/89975216/fbdf466b-0b33-4c6b-a244-c57194895dfe)

## Task 1 :Define custom bin edges based on the maximum values from both y_train and y_test
Show the results function in Task 1:
```
import pandas as pd
import numpy as np

bin_edges = [10, 12, 13, 14, 16.5]  # Adjust these as needed

y_train_cut = pd.cut(y_train, bins=bin_edges, labels=[0, 1, 2, 3], ordered=False)
y_test_cut = pd.cut(y_test, bins=bin_edges, labels=[0, 1, 2, 3], ordered=False)

```
## Task 2 :Define function accuracy, recall, precision and F1 Score
Show the results function in Task 2:
```
def accuracy(yhat, ytrue) :
    correct = 0
    for idx, v in enumerate(ytrue) :
        if v == yhat[idx] :
            correct += 1
    return  correct/len(yhat)   
```
```
def precision(yhat, ytrue, class_det) :
    tp = 0
    fp = 0
    for idx , a in enumerate(ytrue) :
        if yhat[idx] == class_det and a == class_det :
            tp += 1
        if yhat[idx] == class_det and a != class_det :
            fp += 1
    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)        
    return precision
```
```
def recall(yhat, ytrue, class_det) :
    tp = 0
    fn = 0
    for idx , a in enumerate(ytrue) :
        if yhat[idx] == class_det and a == class_det :
            tp += 1
        if yhat[idx] != class_det and a == class_det :
            fn += 1
        if tp+fn == 0:
            recall = 0
        else :
            recall = tp/(tp+fn)

    return recall
```  
def f1(prec, rec):
    if prec == 0 or rec == 0:
        return 0  # Handle the case where either precision or recall is zero
    return (2 * prec * rec) / (prec + rec)   
 ```   
## Task 3 :Deployment 
- Develop a web-based application that contains the model. Here you will be tasked to self-study how to deploy the model into production. Here are some guidelines:
- Here you have multiple options. Those who are veteran web developer may prefer their own web app
stack which is welcomed.
  - Users enter the domain on their browser. They land on your page.
  - (optional) Users may need to navigate to a prediction page.
  - Users read the instruction given on the page that instructs them on how the prediction works.
  - Users find the input form, put in the appropriate data, and click submit.
  - Note that if users do not have information on certain field, you have to allow users to skip that field.In that case, we recommend you to fill the missing field with imputation technique you have learned in the class.
  - A moment later (depending on your model and hardware performance), the result is returned and
printed below the form.

## Results :How to use Applications
This is the working window of the Application you will notice. upper right corner There are 5 details: main page, homework details page, results page, question page, contact page. And the last page is External.

![image](https://github.com/Tonpattra/Machine-Learning/assets/89975216/787550fc-08df-4786-bdab-2bf91f4bfdbf)

The results for Predicting Car Prices are available on the results page.
You can add data 3 type (power, engine and years)

![image](https://github.com/Tonpattra/Machine-Learning/assets/89975216/31abbfa9-d2f2-423b-b62b-32b4267619d7)

After that you can click send, Get the Data Received

![image](https://github.com/Tonpattra/Machine-Learning/blob/main/Homework/HW2/results.png)

And when you need to come back homepage, you cilck Button Back to homework
But you forgot to fill in one of the fields. there is no need to worry Because we will take the middle value of the data to Predict instead.

## Conclusion :
- I tried to write an application for the first time. I really like it and I feel very understanding in class. Thank you for the homework this time. That makes it a good experience!!
- How the new model is better than the old one Ans: I don't know why becuses my model the old is better than the new one TT


