#applied exercise 8
rm(list=ls())
require(ISLR)
require(tree)
library(ISLR)
attach(Carseats)
library(tree)
#8a
set.seed(42)
train <- sample(1:nrow(Carseats),nrow(Carseats)/2)

Carseats.train = Carseats[train,]
Carseats.test = Carseats[-train,] 

#8b
tree.carseats <- tree(Sales ~ ., data = Carseats.train)
summary(tree.carseats)

plot(tree.carseats)
text(tree.carseats, pretty = 0)
#Having 18 nodes makes the tree structure hard to interperate 
pred.carseats = predict(tree.carseats, Carseats.test)
MSE <- mean((Carseats.test$Sales - pred.carseats)^2)
MSE
#the mse is 5.68
#8c
cv.carseats <- cv.tree(tree.carseats)
names(cv.carseats)
plot(cv.carseats$size, cv.carseats$dev, type = "b")

prune.carseats <- prune.tree(tree.carseats, best = 9)

plot(prune.carseats)
text(prune.carseats, pretty = 0)

y_hat <- predict(prune.carseats, newdata = Carseats.test)
prune.MSE <- mean((y_hat - Carseats.test$Sales)^2)
print(prune.MSE)
#yes it did improve 

#8d
library(randomForest)
carseats.bag <- randomForest(Sales ~ ., data = Carseats.train, mtry = 10, ntree = 500, importance = TRUE)
y_hat <- predict(carseats.bag, newdata = Carseats.test)
mse.bag <- mean((Carseats.test$Sales - y_hat)^2)
mse.bag
#the MSE is 2.36
importance(carseats.bag)
#the important variables are ShelveLoc and Price

#8e

carseats.rf <- randomForest(Sales ~ ., data = Carseats.train, mtry = 10/3, ntree = 500, importance = TRUE)
rf.pred <- predict(carseats.rf, Carseats.test)
mse.forr <-mean((Carseats.test$Sales - rf.pred)^2)
mse.forr
plot(carseats.rf)
#the mse is 2.87
importance(carseats.rf)
#The two most important variables (ShelveLoc and Price) stay the same. By having a smaller mtry here it increased our mse by approxmitly 0.5
