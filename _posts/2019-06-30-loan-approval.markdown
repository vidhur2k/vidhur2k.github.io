---
layout: post
title: Classifying Loans based on the risk of defaulting
date: 2019-06-30 07:59:00
---

# A Short Introduction  
Classification is one of the classical problems in Supervised Learning where we attempt to train a model to classify data points into *n* distinct classes. As I was browsing through datasets online, I came across one that contained information on 1000 loan applicants (from both urban and rural areas). One of the columns in the data table was whether or not the loan was approved. An idea immediately struck me:

> What if we could build a model to predict whether an applicant's loan would be approved or denied depending on his or her risk of defaulting?

This would be a garden-variety classification problem, where we have 2 distinct classes to group our data by: a loan approval or a loan denial.  

It is important to not be hasty and start training models on the raw and unexplored data. Preprocessing the data not only helps us smooth out inconsistencies (missing values and outliers), but also gives us a comprehensive understanding of the data which in turn aids us in our model selection process. 

This end-to-end Machine Learning project is primarily based on Python. I have used the following libraries to help me achieve the objective:

1. **Numpy** for mathematical operations.
2. **Pandas** for data exploration and analysis
3. **Matplotlib** and **Seaborn** for data visualization
4. **Scikit-learn** for model training, cross-validation, and evaluation metrics.

# Importing the libraries
Let us perform all the necessary imports beforehand

{% highlight python %}
import numpy as np
np.seterr(divide='ignore')
import math
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
{% endhighlight %}

Once we have all the libraries necessary, we can read the data in from CSV file using Pandas.

{% highlight python %}
data = pd.read_csv('credit_risk.csv')
{% endhighlight %}


# Understanding the Features
Before moving forward with the data exploration, I always like to understand the features that 
I will be dealing with on a superficial level. Doing this will help us put into words any mathematical
interpretations we make. The following are the list of features that we have from our dataset:

1. **Loan ID**: The ID given by the bank to the loan request.
2. **Gender**: The gender of the primary applicant.
3. **Married**: Binary variable indicating the marital status of the primary applicant.
4. **Dependents**: Number of dependents of the primary applicant.
5. **Education**: Binary variable indicating whether or not the primary applicant has graduated high school.
6. **Self_Employed**: Binary variable indicating whether or not the individual is self-employed.
7. **Applicant Income**: The income of the primary applicant.
8. **Co-Applicant Income**: The income of the co-applicant.
9. **Loan Amount**: The amount the applicant wants to borrow.
10. **Loan Amount Term**: The term over which the applicant would repay the loan.
11. **Credit History**: Binary variable representing whether the client had a good history or a bad history.
12. **Loan Status**: Variable indicating whether the loan was approved or denied. This will be our output (dependent) variable.

# Visualizing the data

# Cleaning the data

# Training the model
Finally, the exciting bit! We have our data prepared, and we shall serve it to our model to devour! The algorithm that I chose for this particular case was Logistic Regression. It is one of the simpler supervised learning algorithms, but has proven to be extremely reliant in a variety of instances.

Before we train the model, we shall utilize Scikit-learn's inbuilt train-test split module to randomly split our dataset into training and testing subsets. We shall split it according to the 80-20 rule (this seems an arbitrary and scientifically ungrounded choice, but it is known to "just work" when it comes to training models). 

Let us begin by instantiating a Logistic Regression object (we will be using scikit-learn's module) and split the dataset in the aforementioned way.

{% highlight python %}
# Liblinear is a solver that is effective for relatively smaller datasets.
lr = LogisticRegression(solver='liblinear')

# We will follow an 80-20 split pattern for our training and test data
X_train,X_test,y_train,y_test = train_test_split(data, y, test_size=0.2, random_state = 0)
{% highlight python %}
{% endhighlight %}
{% endhighlight %}

Now that we have everything we need, we fit the model to the training data.

{% highlight python %}
lr.fit(X_train, y_train)
{% endhighlight %}

# Evaluating the performance of the model

Now that the model has been trained, we will use the test data that we sieved from the original dataset to evaluate how well our model generalizes to the data. I have divided the evaluation process in the following way:

1. Vectorize the predictions made by the model and build a confusion matrix.
2. Use the confusion matrix 

{% highlight python %}
# We will compare this vector of predictions to the actual target vector to determine the model performance.
y_pred = lr.predict(X_test)

# Build the confusion matrix.
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
class_names=[0,1] # name of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# The heatmap requires that we pass in a dataframe as the argument
sns.heatmap(pd.DataFrame(confusion_matrix), annot=True, cmap="YlGnBu", fmt="g")

# Configure the heatmap parameters
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
{% endhighlight %}