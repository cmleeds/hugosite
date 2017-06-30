+++
date = "2017-06-29T18:06:57-05:00"
description = "test"
title = "Suriviving the Titanic"
draft = "True"
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
print(train0.shape,test0.shape)

(891, 12) (418, 11)
{{< /highlight  >}}

After reviewing the variable descriptions, it seems like the `Ticket` & `Name` variables may not be very useful. 

{{< highlight python >}}
print(len(train0.Ticket.unique()),len(train0.Name.unique()))

681 891
{{< /highlight >}}


As expected, each value of `Name` is unique and there is not a whole lot of replication for the each `Ticket` values. But maybe we can extract some meaningful information from the raw values. 

Other kernels on kaggle seem to suggest that we can extract some useful information from name by taking the title out of the names. A 'title' variable would probably get's it's value from being proxy for sex and social class. 


<img align="left" src="/img/titanicbarplot1.svg" width="50%" height="50%"/>
<img align="right" src="/img/titanicbarplot2.svg" width="50%" height="50%"/>




 more text after this plot.
