+++
date = "2017-06-29T18:06:57-05:00"
description = "test"
title = "Suriviving the Titanic"
draft = "False"
+++

## Introduction

One of the most popular data science 'competitions' on [kaggle](https://www.kaggle.com/) asks data scientists to use passenger information from the RMS [titanic](https://www.kaggle.com/c/titanic) in order to predict survival rate. A description of the data can be found [here](https://www.kaggle.com/c/titanic/data).

## Libraries

As always, Let's start be bringing in our favorite libraries required for analysis.

{{< highlight python >}}

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

{{< /highlight >}}

## Import Data

`pandas` allows us to easily bring our data in and take a look at what is inside of it.  

{{< highlight python >}}
train0 = pd.read_csv("train.csv")
test0 = pd.read_csv("test.csv")
{{< /highlight >}}

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

After reviewing the variable descriptions, it seems like the `Ticket` & `Name` variables may not be very useful.

Other kernels on kaggle seem to suggest that we can extract some useful information from name by taking the title out of the names. A 'title' variable would probably get's it's value from being proxy for sex and social class.

**But**, Let's explore some of our other features before extracting new features.

<img align="left" src="/img/titanicbarplot1.svg" width="50%" height="40%"/>
<img align="right" src="/img/titanicbarplot2.svg" width="50%" height="40%"/>

<img align="left" src="/img/titanicdensity1.svg" width="50%" height="40%"/>
<img align="right" src="/img/titanicdensity2.svg" width="50%" height="40%"/>

.

From the plots above, we can see that sex will be a large contributor to survival rate. `Pclass` & `Embarked` might have some effect. However, there does not seem to be a visible difference between `Age` & `Fare` densities when we plot by survival rate.

## Feature Engineering

Let's move ahead and extract some meaning from our `Name` & `Cabin` variables. First, let's combine our test and training data to ensure that we process them in the same way.

{{< highlight python >}}
datalist = [train0,test0]
alldata = pd.concat(datalist)
alldata.shape
{{< /highlight  >}}

```
(1309, 12)
```


While each name is a unique string of letters, with the `str.extract` functionality from the `pandas` library, we can extract the first occurance of a regular expression pattern which will greatly reduce the manual that would otherwise be needed.

{{< highlight python >}}
alldata['Salutation'] = alldata.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
alldata.groupby('Salutation').size()
{{< /highlight  >}}

```
Salutation
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

This will help us gather all *titles* that occur before the '.' in each Name.
alter rare titles to 'Rare' value and remove unneeded columns.

{{< highlight python >}}
alldata['title'] = alldata['title'].replace(['Lady','Countess','Capt','Col'\
,'Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],'Rare')
alldata.drop(["Ticket","Name","Cabin"],1,inplace=True)
{{< /highlight  >}}


{{< highlight python >}}
train1 = alldata[alldata.Survived.isnull()]
test1 = alldata[~alldata.Survived.isnull()]
{{< /highlight  >}}
