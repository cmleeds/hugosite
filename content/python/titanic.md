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
%matplotlib inline
{{< /highlight >}}

## Import Data

`pandas` allows us to easily bring our data in and take a look at what is inside of it.  

{{< highlight python >}}
train0 = pd.read_csv("train.csv")
test0 = pd.read_csv("test.csv")
{{< /highlight >}}

## Exploring our Data

> "You can see a lot just by observing." - Yogi Berra

We can see the description of our data with variables notes [here](https://www.kaggle.com/c/titanic/data). 

{{< highlight python >}}
print(train0.columns.values)

['PassengerId' 'Survived' 'Pclass' 'Name' 'Sex' 'Age' 'SibSp' 'Parch'
 'Ticket' 'Fare' 'Cabin' 'Embarked']
{{< /highlight  >}}

After reviewing the variable descriptions, it seems like the `Ticket` & `Name` variables may not be very useful. 

Other kernels on kaggle seem to suggest that we can extract some useful information from name by taking the title out of the names. A 'title' variable would probably get's it's value from being proxy for sex and social class. But, Let's explore some of our other features before extracting new features. 

<img align="left" src="/img/titanicbarplot1.svg" width="50%" height="40%"/>
<img align="right" src="/img/titanicbarplot2.svg" width="50%" height="40%"/>

<img align="left" src="/img/titanicdensity1.svg" width="50%" height="40%"/>
<img align="right" src="/img/titanicdensity2.svg" width="50%" height="40%"/>

.

From the plots above, we can see that sex will be a large contributor to survival rate. `Pclass` & `Embarked` might have some effect. However, there does not seem to be a visible difference between `Age` & `Fare` densities when we plot by survival rate. 


