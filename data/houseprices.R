# load libraries
library(dplyr)
library(tidyr)
library(ggplot2)

# import data
train0 <- read.csv("house_prices_train.csv",stringsAsFactors = F)
test0 <- read.csv("house_prices_test.csv",stringsAsFactors = F)

# what is in the data?
train0 %>% glimpse()

# how many missing values per variable?
train0 %>% 
  summarise_all(funs(100 * mean(is.na(.)))) %>% 
  gather(feature,percent_missing) %>% 
  filter(percent_missing!=0) %>%
  ggplot(aes(x=reorder(feature,percent_missing),
             y=percent_missing)) +
  geom_bar(stat="identity",fill="skyblue") +
  theme(axis.text.x = element_text(angle=90)) +
  coord_flip() +
  theme_light() +
  labs(x="Features with Missing Values",
       y="% Missing")


# plot training sale price
train0 %>% 
  ggplot(aes(x=SalePrice)) +
  geom_histogram(fill="skyblue",aes(y = (..count..)/sum(..count..))) +
  theme_light() +
  scale_x_continuous(labels = scales::dollar) +
  scale_y_continuous(labels = scales::percent) +
  labs(x="Sale Price",
       y="")


train0 %>% 
  group_by(PoolQC,PoolArea) %>%
  summarise(n = n(),
            mean_price = mean(SalePrice,na.rm=T))


train0 %>% 
  group_by(MiscFeature) %>%
  summarise(n = n(),
            mean_price = mean(SalePrice,na.rm=T))
