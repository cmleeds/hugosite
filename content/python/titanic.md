+++
date = "2017-06-29T18:06:57-05:00"
description = "test"
title = "Surviving the Titanic"
draft = "False"
+++

<hr>
## Introduction

One of the most popular data science competitions on [kaggle](https://www.kaggle.com/) asks users to predict survival from passenger information from the RMS [titanic](https://www.kaggle.com/c/titanic). A description of the data can be found [here](https://www.kaggle.com/c/titanic/data).

We want to be able to classify any individual passenger as 1='Survived' or 0='Not Survived' based on the other features in the dataset. Obviously, predicting survival rate on the titanic provides no practical application for future use. But, models can also provide information on the importance of the inputs(features). In this way, we can assess the relative contribution of any individual characteristic in predicting survival.

<hr>
## Libraries


As always, Let's start be bringing in our favorite libraries required for data wrangling & analysis. We need `pandas` and `numpy` for wrangling and processing, `seaborn` & `matplotlib` for plotting, & `sklearn` for analytics.

{{< highlight python >}}

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# plotting options
%matplotlib inline
sns.set_style('whitegrid')

# for model tuning & fitting
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

{{< /highlight >}}

We are going to use the `GridSearchCV` function to optimize a our `GradientBoostingClassifier`. By estimating model accuracy over a grid of possible model parameters, we can select the best choice from amongst a series of gradient boosted classifiers.

<hr>
## Import Data

we can use the pandas `read_csv` function to read in our datasets for processing.

{{< highlight python >}}
train0 = pd.read_csv("train.csv")
test0 = pd.read_csv("test.csv")
{{< /highlight >}}

I find it best to keep tabs on each *version* of our dataset through process and prevent overwriting. So, we start at version-0 for each dataset, (e.g. `test0` & `train0`).

<hr>
## Exploring our Data

> "You can see a lot just by observing." - Yogi Berra

We can see the description of our data with variables notes [here](https://www.kaggle.com/c/titanic/data). Our training and testing datasets contain passenger level data where we want to predict on `Survived` where 1 = *Survived* and 0 = *Did not Survive*.

{{< highlight python >}}
train0.info()
{{< /highlight  >}}


```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            714 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          204 non-null object
Embarked       889 non-null object
dtypes: float64(2), int64(5), object(5)
memory usage: 83.6+ KB
```

We have missing data in Age, Cabin & Embarked. Also, we'll need to convert each of of our character variables to factors before modeling fitting with `sklearn`.

After reviewing the variable descriptions, it seems like the `Ticket` & `Name` variables may not be very useful. Other kernels on kaggle seem to suggest that we can extract some useful information from name by taking the title out of the names. A 'title' variable would probably get's it's value from being proxy for sex and social class.

**But**, Let's explore some of our other features before extracting new features. Below are some basic plots to explore the relationships between gender, ticket class, port of embarktion, age, & fare against survival rate.

<img align="left" src="/img/titanicbarplot1.svg" width="50%" height="40%"/>
<img align="right" src="/img/titanicbarplot2.svg" width="50%" height="40%"/>
<img align="left" src="/img/titanicdensity1.svg" width="50%" height="40%"/>
<img align="right" src="/img/titanicdensity2.svg" width="50%" height="40%"/>

.

From the plots above, we can see that sex will be a large contributor to survival rate. Also, odds of survival increase with ticket class. Futhermore, passengers that embarked at port C survived seem to have higher survival rates. However, there does not seem to be a visible difference between `Age` & `Fare` densities with respect to our response.

<hr>
## Feature Engineering

Let's move ahead and extract some meaning from our `Name` & `Cabin` variables. First, let's combine our test and training datasets. We should do this to ensure we are processing the data in the same way.

{{< highlight python >}}
datalist = [train0,test0]
alldata = pd.concat(datalist)
alldata.shape
{{< /highlight  >}}

```
(1309, 12)
```

We can utilize the `str.extract` functionality from the `pandas` library to extract the first occurrence of a regular expression pattern. This reduces the time needed to manually inspect and extract each title from the `Name` feature.

{{< highlight python >}}
alldata['title'] = alldata.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
alldata.groupby('title').size()
{{< /highlight  >}}

```
title
Capt          1
Col           4
Countess      1
Don           1
Dona          1
Dr            8
Jonkheer      1
Lady          1
Major         2
Master       61
Miss        260
Mlle          2
Mme           1
Mr          757
Mrs         197
Ms            2
Rev           8
Sir           1
dtype: int64
```

After extracting our *titles*, let's 'bin' the less common titles into a *Rare* category. Then, let's use the seaborn package to see if our new feature may have some value in modeling.

{{< highlight python >}}

alldata['title'] = alldata['title'].replace(['Mme','Lady','Countess','Capt','Col'\
,'Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],'Rare')

plotthis = alldata1[~alldata1.Survived.isnull()]
plot = sns.barplot(x="title",y="Survived",data=plotthis);
plt.xlabel('Passenger Title')
plt.ylabel('Mean Survival Rate')

{{< /highlight  >}}


![](/img/titanicbarplot3.svg)

We can see significant differences in the survival rate amongst our new feature. This plot suggests that title may provide some value as we craft our models.

Finally, let's transform our character features into factors for the model fitting process & change the null values.

{{< highlight python >}}

# change to factors
alldata1['title'] = alldata1['title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5})
alldata1['Sex'] = alldata1['Sex'].map({"male": 1, "Female": 2})
alldata1["Embarked"] = alldata1['Embarked'].map({"S": 1, "C": 2, "Q": 3})

# change missing values to 0
alldata1['title'] = alldata1['title'].fillna(0)
alldata1['Sex'] = alldata1['Sex'].fillna(0)
alldata1['Embarked'] = alldata1['Embarked'].fillna(0)
alldata1['Age'] = alldata1['Age'].fillna(0)
alldata1['Fare'] = alldata1['Fare'].fillna(0)

{{< /highlight  >}}

<hr>
## Model Tuning

Now that we are through the wrangling, exploration and feature engineering phases, we can train our models. First, we split our data apart for training and testing.

{{< highlight python >}}
# split test and train
train1 = alldata1[~alldata1.Survived.isnull()].drop("PassengerId",1)
test1 = alldata1[alldata1.Survived.isnull()]

# split features and response
X = train1.drop("Survived",1)
Y = train1["Survived"]
testX = test1.drop(['Survived','PassengerId'],1)
{{< /highlight  >}}



We want to search for the best model fit over a range of parameters. When performing a grid search for optimal parameter values, we should be aware that this procedure is very computationally intensive. In essence, we are fitting the same type of model many times over a 3-fold cross-validation for each combination of parameters and selecting parameters with that provide the best accuracy score.

{{< highlight python >}}

params = {'n_estimators':range(300,1201,100),
         'learning_rate':[0.1,0.05,0.02],
         'max_depth':[2,3],
         'max_features':[3,4,5]}

grdsearch = GridSearchCV(estimator=GradientBoostingClassifier(),
                         param_grid=params,
                         scoring='accuracy',
                         iid=False,
                         cv=3)

searchresults = grdsearch.fit(X,Y)

{{< /highlight  >}}

once the grid search is complete, we can inspect the attributes of our fit object `searchresults` for the optimal parameters for prediction.

{{< highlight python >}}
print("Best score acheived is:",searchresults.best_score_)
print("Best grid parameters for this score are:")
print(searchresults.best_params_)
{{< /highlight  >}}

```
Best score acheived is: 0.842873176207
Best grid parameters for this score are:
{'learning_rate': 0.1, 'max_depth': 3, 'max_features': 3, 'n_estimators': 300}
```

Since we are utilizing 3-fold cross-validation, we can avoid selecting the best model based on the training error. This is a useful measure to avoid over-fitting.

<hr>
## Final Prediction

Now that we have our parameters tuned from the grid search, we can extract predictions from our model fit and output the results for submission to kaggle. We need to do this in order to get the our test error rate.

{{< highlight python >}}
gbc = GradientBoostingClassifier(n_estimators=300,
                                 learning_rate=0.1,
                                 max_depth=3,
                                 max_features=3)

model = gbc.fit(X,Y)
survival_prediction = model.predict(testX)
survival_id = test1['PassengerId']
{{< /highlight  >}}

The gradient boosting model has a `feature_importances_` attribute which we can extract for plotting with column names. Below is a barplot of the relative importance for each of our features.

![](/img/titanicImportance.svg)

Finally, we can use the `to_csv` functionality to output or predictions on the test set and submit the results for review.

{{< highlight python >}}
submitFile = pd.DataFrame({ 'PassengerId': survival_id,
                            'Survived': survival_prediction })
submitFile.to_csv("titanicsubmission.csv", index=False)
{{< /highlight  >}}

When submitting this file to kaggle for submission, the test error rate equals **0.77512**.
