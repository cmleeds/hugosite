+++
date = "2017-09-04"
description = "test"
title = "Predicting House Prices"
draft = "True"
+++

<hr>
## Introduction

[Kaggle](https://www.kaggle.com/) provides a few *getting started* competitions for the machine learning beginner. The competition we will explore [here](https://www.kaggle.com/c/house-prices-advanced-regression-techniques), asks data scientists to predict home prices in Ames, Iowa using advanced regressions techniques. A code book with variable descriptions of the data can be found [here](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data).



<hr>
## Libraries

There are a handful of libraries in R that are optimized for manipulation of large datasets. `data.table` & `dplyr` will be used to speed up and enhance data-manipulation. As we will soon see, dplyr works well with `ggplot2` and has a unique syntax with the **%>%** operator which makes for easier data wrangling in a interactive programming environment. Instead of writing code in the order of operations, we can 'pipe' operations from left to right and top to bottom. Personally, I think this makes for an easier and more natural programming style (in an interactive environment, specifically).

{{< highlight r >}}

library(data.table)
library(dplyr)
library(ggplot2)
library(scales)
library(xgboost)
library(caret)
library(tidyr)

{{< /highlight >}}

`caret` & `xgboost` will provide missing value imputation capability and allow us to perform some model optimization with grid search capabilities and cross-validation.

While we can use this as an opportunity to explore other regularized regression techniques, xgboost provides a framework for regularization. And, we will attempt to explore this method in this post given it's popularity.

<hr>
## Import Data

data.table's *fast and friendly file finagler* (`data.table::fread`) brings our data into the workspace quickly and efficiently.

{{< highlight r >}}
train0 <- fread("house_prices_train.csv")
test0 <- fread("house_prices_test.csv")
test0[,SalePrice := NA] # add response column to test data

# merge datasets for pre-processing
all_data0 <- rbindlist(list(train0,test0))

# how much data?
all_data0 %>% dim()

{{< /highlight >}}

By binding our datasets together, we can ensure that we process the training and test sets in the same way. the resulting dataframe has 1459 observations with 80 training features. (81 features including our response *SalePrice*)

{{< highlight rconsole >}}
2919   81
{{< /highlight  >}}

<hr>
## Feature Engineering

First, let's get a a sense of how severe the missing values problem is for our data.

{{< highlight r >}}
# count missing data per each variable
missing_data <- melt(all_data0[,
                               lapply(.SD, function(x) sum(is.na(x)))
                               ])
# plot missing data
missing_data %>%
  filter(value > 0) %>%
  filter(variable != "SalePrice") %>%
  ggplot(aes(x = reorder(variable, value), y = value)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  theme_light() +
  theme(axis.text.x = element_text(angle = 90)) +
  coord_flip() +
  labs(x = "",
       y = "# of missing values")
{{< /highlight >}}

<img align="center" src="/img/house_reg_missing_vals.svg" width="80%" height="20%"/>

We can see that some of our features have significant missing value problems. However, most features with missing values do not have a significant problem. By looking through the code book, many of the features with missing values are categorical. Let's take a look at some of the most severe issues.

{{< highlight r >}}
all_data0 %>%
  ggplot(aes(y=SalePrice,x=PoolQC)) +
  geom_boxplot() +
  theme_light() +
  scale_y_continuous(labels = scales::comma_format()) -> p1

all_data0 %>%
  ggplot(aes(y=SalePrice,x=Alley)) +
  geom_boxplot() +
  theme_light() +
  scale_y_continuous(labels = scales::comma_format()) -> p2

all_data0 %>%
  ggplot(aes(y=SalePrice,x=Fence)) +
  geom_boxplot() +
  theme_light() +
  scale_y_continuous(labels = scales::comma_format()) -> p3

all_data0 %>%
  ggplot(aes(y=SalePrice,x=MiscFeature)) +
  geom_boxplot() +
  theme_light() +
  scale_y_continuous(labels = scales::comma_format()) -> p4


require(gridExtra)
grid.arrange(p1,p2,p3,p4)
{{< /highlight >}}


<img align="center" src="/img/some_missing_features.svg" width="70%" height="20%"/>

It seems that some of our categorical features with missing data may provide some values by including our these features. Let's change our *NA* values to *none*.

{{< highlight r >}}

{{< /highlight >}}
