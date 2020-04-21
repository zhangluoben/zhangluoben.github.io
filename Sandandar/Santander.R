rm(list=ls())
library(tidyverse)
library(xgboost)
library(magrittr)
library(moments)
library(caret)
library(irlba)

set.seed(0)

################################
######    Read in Data    ######
################################
train <- read_csv("train.csv")
test <- read_csv("test.csv")

################################
######    Initial Clean   ######
################################
# remove duplicates common function
find_duplicates <- function(x) 
  lapply(x, c) %>% # will return boolean of duplicated and we will filter out later
  duplicated %>% 
  which 

# remove highly correlated variables
find_correlations <- function(x) 
  data.matrix(x) %>% # compute correaltions and will filter out later
  cor(method = "spearman") %>% 
  findCorrelation(cutoff = 0.98)

# get rid of column with 0's
zero_variance <- function(x) # boolean check on variance and will filter out later
  var(x) != 0

################################
######    Build Model     ######
################################
# set up XGBoost model through a function
get_xgb <- function(num_class = 2, nfold = 5) {
  set.seed(0)
  p <- list(tree_method = "hist", # needed argument for grow_policy arg "lossguide"
            grow_policy = "lossguide", # splits at nodes with highest loss change
            objective = "multi:softprob", # since we specificy num_class
            eval_metric = "mlogloss", # our competition's metric
            num_class = num_class, # specified in the function
            nthread = 8, # a CPU thread indicated that was successful in other models
            eta = 0.15, # a learning rate parameter
            max_depth = 0, # needed for "lossguide"
            min_child_weight = 5, # needed weight parameter of a child leaf in order to partition 
            max_leaves = 14, # self-explanatory
            subsample = 0.5, # partitioning of training data
            colsample_bytree = sqrt(ncol(train_test)) / ncol(train_test)) # subsample ratio of columns when constructing each tree
  
  dtrain <- xgb.DMatrix(data = data.matrix(train_test[train_vec, ]), 
                        label = as.integer(cut(target, num_class)) - 1) # read in training data
  
  dtest <- xgb.DMatrix(data = data.matrix(train_test[-train_vec, ])) # read in testing data
  
  cv <- xgb.cv(p, dtrain, print_every_n = 50, prediction = TRUE, nfold = nfold,
               early_stopping_rounds = 50, nrounds = 500) # cross validate to find best tree
  nrounds <- round(cv$best_iteration * (1 + 1 / nfold))
  
  m_xgb <- xgb.train(p, dtrain, nrounds, verbose = -1) # training best tree
  
  pred <- rbind(cv$pred, predict(m_xgb, dtest, reshape = TRUE)) %>% 
    as_tibble() %>% # dataframe sub class commonly used in competitions for speed and iterability in R ecosystem
    mutate(which_class = apply(., 1, which.max)) %>% 
    set_names(paste0("xgb", num_class, "_", 1:ncol(.))) # predictions using best tree
}

################################
######    Cleaning Data    #####
################################
train_vec <- 1:nrow(train) # vector to be used later
target <- log1p(train$target) # logarithmic computation for targets

train %>% 
  select(-ID, -target) %>%
  select_if(zero_variance) %>% # first data cleaning function
  select(-find_duplicates(.)) %>% # second data cleaning function
  select(-find_correlations(.)) %>% # third data cleaning function
  bind_rows(select(test, names(.))) ->
  train_test

rm(train, test); gc() # garbage collection for removed sparse data frames


################################
###    Model Construction    ###
################################
# creating XGBoost features
k <- 2:6
train_test_xgb <- data.frame(row.names = 1:nrow(train_test))
for (i in seq_along(k)) { # for each iteration, we call the get_xgb function to retrieve both train and test metrics
  cat("k =", k[i], "\n")  # and stop when it finds the best result within the get_xgb function for each k value
  train_test_xgb <- cbind(train_test_xgb, get_xgb(k[i]))
  gc()
}

#Binding features with necessary metrics to allow us to minimize competition metric
train_test %<>% # update and reassign
  mutate(mean = apply(train_test, 1, mean),
         gmean = apply(train_test, 1, function(x) expm1(mean(log1p(x)))),
         sd = apply(train_test, 1, sd),
         max = apply(train_test, 1, max),
         kurt = apply(train_test, 1, kurtosis),
         skew = apply(train_test, 1, skewness),
         row_sum = apply(train_test, 1, sum),
         iqr = apply(train_test, 1, IQR),
         n_uniq = apply(train_test, 1, n_distinct),
         n_zeros = apply(train_test, 1, function(x) sum(x == 0)),
         zero_frac = n_zeros / ncol(train_test)) %>% 
  bind_cols(train_test_xgb) %>% 
  data.matrix()

# Preparing data for final construction
dtest <- xgb.DMatrix(data = train_test[-train_vec, ]) # test set
train_test <- train_test[train_vec, ] # all of the data
train_vec <- createDataPartition(target, p = 0.9, list = F) %>% c() # partition for train/test
dtrain <- xgb.DMatrix(data = train_test[train_vec, ], label = target[train_vec]) # y train
dval <- xgb.DMatrix(data = train_test[-train_vec, ], label = target[-train_vec]) # y test
cols <- colnames(train_test) # used to plot importance

rm(train_test, target, train_vec); gc() # remove unecessary data

################################
#####    Model Training    #####
################################

param <- list(objective = "reg:linear",
          booster = "gbtree",
          eval_metric = "rmse",
          nthread = 8,
          eta = 0.1,
          max_depth = 30,
          min_child_weight = 100,
          gamma = 10,
          subsample = 0.5,
          colsample_bytree = 0.3,
          colsample_bylevel = 0.3,
          alpha = 0,
          lambda = 760,
          nrounds = 10000)

m_xgb <- xgb.train(param, dtrain, param$nrounds, list(val = dval), print_every_n = 100, early_stopping_rounds = 700)

xgb.importance(cols, model = m_xgb) %>% 
  xgb.plot.importance(top_n = 30)

################################
######    Write to CSV    ######
################################

read_csv("sample_submission.csv") %>%  
  mutate(target = expm1(predict(m_xgb, dtest))) %>%
  write_csv(paste0("xgb_aec_", round(m_xgb$best_score, 5), ".csv"))
