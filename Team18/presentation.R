#install.packages('randomForest')
install.packages('rmdformat')
rm(list=ls())
library(randomForest)
library(MASS)
set.seed(1)
head(Boston)
dim(Boston)

#Bagging
train <-  sample(1:nrow(Boston), nrow(Boston)/2)
boston.test=Boston[-train ,"medv"]
#Bagging is only a special case of random forest. mtry=13 mean 13 predictors is used
bag.boston=randomForest(medv~.,data=Boston, subset=train, mtry=13,importance =TRUE)
bag.boston

yhat.bag = predict(bag.boston ,newdata=Boston[-train ,])
plot(yhat.bag, boston.test)
abline(0,1)
mean((yhat.bag-boston.test)^2)

bag.boston=randomForest(medv~.,data=Boston ,subset=train,
                        mtry=13,ntree=25)
yhat.bag = predict(bag.boston ,newdata=Boston[-train ,])
mean((yhat.bag-boston.test)^2)
                   
#Random Forest
set.seed(1)
rf.boston <- randomForest(medv~.,data=Boston ,subset=train ,
                         mtry=6,importance =TRUE)
yhat.rf <- predict(rf.boston ,newdata=Boston[-train ,])
mean((yhat.rf-boston.test)^2)

importance (rf.boston)

varImpPlot (rf.boston)

library(gbm)
set.seed(1)
boost.boston=gbm(medv~.,data=Boston[train,],distribution="gaussian",n.trees=5000,interaction.depth=4)

summary(boost.boston)

par(mfrow=c(1,2))
plot(boost.boston,i="rm")
plot(boost.boston,i="lstat")

yhat.boost=predict(boost.boston,newdata=Boston[-train,],n.trees=5000)
mean((yhat.boost-boston.test)^2)

boost.boston=gbm(medv~.,data=Boston[train,],distribution="gaussian",n.trees=5000,interaction.depth=4,shrinkage=0.2,verbose=F)
yhat.boost=predict(boost.boston,newdata=Boston[-train,],n.trees=5000)
mean((yhat.boost-boston.test)^2)