# load libraries
library(dplyr)
library(MCMCpack)

# load data
data <- datasets::iris

# what's in the data?
data %>% glimpse()

index <- sample(1:nrow(data),75,replace=F)
data_old <- data[index,]
data_new <- data[!index,]

model1 <- aov(Sepal.Width ~ Species,data=data_old)
summary(model1)
coef(model1)

data_old %>%
  group_by(Species) %>%
  summarise(stderr = var(Sepal.Width,na.rm=T))
