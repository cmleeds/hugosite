+++
date = "2017-06-29"
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

{{< highlight python "style=monokai">}}
# data wrangling and exploration
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set_style('whitegrid')

# data processing
from sklearn.preprocessing import Imputer

# for model tuning & fitting
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
{{< /highlight >}}

We are going to use the `GridSearchCV` function to optimize a our `RandomForestClassifier`. By estimating model accuracy over a grid of possible model parameters & data partitions, we can optimize our hyper-parameter selection.

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


{{< highlight console >}}
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
{{< /highlight  >}}

We have missing data in Age, Cabin & Embarked. We should impute missing values for these observations in a way that won't impact the results but will allow us to leverage the information from the other features present in those observations.  

Furthermore, if we see value in retaining our non-numeric features, then we'll need to transform these into dummy variables.

After reviewing the variable descriptions, it seems like the `Ticket`, `Name` & `Cabin` variables may not be very useful in their current form. Let's aim to extract some useful information from these features. We can extract the title and cabin group information from these variables.

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

{{< highlight console >}}
(1309, 12)
{{< /highlight  >}}

We can utilize the `str.extract` functionality from the `pandas` library to extract the first occurrence of a regular expression pattern. This reduces the time needed to manually inspect and extract each title from the `Name` feature.

{{< highlight python >}}
alldata['title'] = alldata.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
alldata.groupby('title').size()
{{< /highlight  >}}

{{< highlight console >}}
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
{{< /highlight  >}}

After extracting our *titles*, let's 'bin' the less common titles into a *Rare* category. Then, let's use the seaborn package to see if our new feature may have some value in modeling.

{{< highlight python >}}
alldata['title'] = alldata['title'].replace(['Mme','Lady','Countess','Capt','Col'\
,'Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],'Rare')
{{< /highlight  >}}

![](/img/titanicbarplot3.svg)

We can see significant differences in the survival rate amongst our new feature. This plot suggests that title may provide some value as we craft our models.

we can extract group information from `Cabin`.

{{< highlight python >}}
alldata0.groupby('CabinGroup').size()
{{< /highlight  >}}

{{< highlight console >}}
CabinGroup
A    22
B    65
C    94
D    46
E    41
F    21
G     5
T     1
dtype: int64
{{< /highlight  >}}


Finally, let's transform our drop our old features and transform our new features into binary dummy variables. This is important to represent our data in the most accurate way possible. While we could transform these features to integers representing each level, this would imply that there is some ordinal aspect to these features. It is important to represent our data in the most accurate way possible.  

{{< highlight python >}}
alldata1 = alldata0.drop(["Ticket", "Name", "Cabin"], 1, )
alldata2 = pd.get_dummies(alldata1, dummy_na = True)
alldata2.head().shape
{{< /highlight  >}}

as we can see, we have expanded the dimensionality of our data very quickly with this one call to `get_dummies`.

{{< highlight console >}}
(5, 31)
{{< /highlight  >}}

Finally, we can address our missing value problem identified from our first looks at the raw data. `Age` and `Fare` have missing values that need to be addressed.

{{< highlight python >}}
alldata2['Survived'] = alldata2['Survived'].fillna(-1) # keep Imputer from altering response data
fill_NaN = Imputer(missing_values = np.nan, strategy = 'median', axis=1)
alldata3 = pd.DataFrame(fill_NaN.fit_transform(alldata2))
alldata3.columns = alldata2.columns
alldata3.index = alldata2.index
{{< /highlight >}}

Let's check to make sure that our imputation has not fundamentally altered our distribution for age as this could increase the bias in our results.


<img align="left" src="/img/age_before_impute.svg" width="50%" height="40%"/>
<img align="right" src="/img/age_after_impute.svg" width="50%" height="40%"/>
 .

<hr>
## Model Tuning

Now that we are through the wrangling, exploration and feature engineering phases, we can train our models. First, we split our data apart for training and testing.

{{< highlight python >}}
# split test and train
train1 = alldata3[alldata3.Survived != -1].drop("PassengerId",1)
test1 = alldata3[alldata3.Survived == -1]

# split features and response
X = train1.drop("Survived",1)
Y = train1["Survived"]
testX = test1.drop(['Survived','PassengerId'],1)
{{< /highlight  >}}



We want to search for the best model fit over a range of parameters. When performing a grid search for optimal parameter values, we should be aware that this procedure is very computationally intensive. In essence, we are fitting the same type of model many times over a 3-fold cross-validation for each combination of parameters and selecting parameters with that provide the best accuracy score.

{{< highlight python >}}
param_grid = {'n_estimators':range(50,501,50),
              "max_depth": [3, 4, 5],
              "max_features": ["sqrt"],
              "min_samples_split": [2, 5, 10],
              "min_samples_leaf": [2, 5, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

grdsearch = GridSearchCV(estimator = RandomForestClassifier(),
                         param_grid = param_grid,
                         scoring = 'accuracy',
                         iid = False,
                         cv = 3)

searchresults = grdsearch.fit(X,Y)
{{< /highlight  >}}

once the grid search is complete, we can inspect the attributes of our fit object `searchresults` for the optimal parameters for prediction.


{{< highlight console >}}
Best score acheived is: 0.826038159371
Best grid parameters for this score are:
{'bootstrap': False, 'criterion': 'entropy', 'max_depth': 5, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 100}
{{< /highlight  >}}

Since we are utilizing 3-fold cross-validation, we can avoid selecting the best model based on the training error. This is a useful measure to avoid over-fitting.

<hr>
## Final Prediction

Now that we have our parameters tuned from the grid search, we can extract predictions from our model fit and output the results for submission to kaggle. We need to do this in order to get the our test error rate.

{{< highlight python >}}
rfmodel = RandomForestClassifier(bootstrap = True,
                                 criterion = "entropy",
                                 max_depth = 5,
                                 max_features = 'sqrt',
                                 min_samples_leaf = 2,
                                 min_samples_split = 2,
                                 n_estimators = 100)

# generate predictions
model = rfmodel.fit(X,Y)
survival_prediction = model.predict(testX).astype('int').astype('str')
survival_id = test1['PassengerId']
{{< /highlight  >}}

The gradient boosting model has a `feature_importances_` attribute which we can extract for plotting with column names. Below is a barplot of the relative importance for each of our features.

![](/img/titanicImportance.svg)

Finally, we can use the `to_csv` functionality to output or predictions on the test set and submit the results for review.

{{< highlight python >}}
submitFile = pd.DataFrame({ 'PassengerId': survival_id,
                            'Survived': survival_prediction })
submitFile.to_csv("titanicsubmission.csv", index = False)
{{< /highlight  >}}

When submitting this file to kaggle for submission, the test accuracy equals **80.861%**!!
