library(dplyr)
library(plyr) #ddply
library(caret)

#------ read features
setwd('/media/diskD/EPFL/Fall 2016/DELA/Project')
db=read.csv('OutputTable.csv', stringsAsFactors = F)

#------ sort submissions
db=db[order(db$UserID,db$ProblemID,db$SubmissionNumber),]

#--- replace NA values with 0
db[is.na(db)]=0

#----- remove first submissions
db= filter(db,SubmissionNumber>0)

#---- remove cases when there is no video or forum activity between two submissions
db$NVideoAndForum= db$NVideoEvents+db$NForumEvents
db= filter(db,NVideoAndForum>0)  

#----- make a categorical variable, indicating if grade improved
db$improved = factor(if_else(db$GradeDiff>0 ,'Yes', 'No' ))
table(db$improved)

write.csv(db, file='OutputTable.csv')
#----- visualize features per each category
boxplot(db$TimeSinceLast ~ db$improved , horizontal = T, outline=F)
boxplot(db$NForumEvents ~ db$improved , horizontal = T, outline=F)
boxplot(db$NVideoEvents ~ db$improved , horizontal = T, outline=F)
boxplot(db$NumberOfThreadViews ~ db$improved , horizontal = T, outline=F)


#============ train a classifier to predict 'improved' status =============

# ----- 1. split data into train and test set
set.seed(1234)
tr.index= sample(nrow(db), nrow(db)*0.6)
db.train= db[tr.index,]
db.test = db[-tr.index,]


#-----
# Train a classifier to identify which features are most predictive
# of an increase versus decrease of the grade. Try different methods, 
# model parameters, features set and find the best classifier (highest 'kappa' value on test set.)
# try to achieve a model with kappa > 0.5 
#

db_features =read.csv('features.csv', stringsAsFactors = F)

# Convert Categorical Variables into integer
db_features$improved = sapply(db_features$improved, switch, "No" = 0, "Yes" = 1,USE.NAMES = F)
View(db_features)


######### REGRESSION MODEL 

# Trying an SVM model to predict overalGradeDiff given avgTbwSubs and CountOfVideoandForumEvents

# ----- 1. Split data into train and test set
set.seed(1234)
tr.index= sample(nrow(db_features), nrow(db_features)*0.6)
db_features.train= db_features[tr.index,]
db_features.test = db_features[-tr.index,]
# ---- 2 . Define the feature set
fs = c('avgTbwSubs','countOfVideoandForumEvents')
svmReg = train(x=db_features.train[,fs],y=db_features.train$overalGradeDiff, method = "svmLinear") 
pred=predict(svmReg, newdata=db_features.test[,fs])
RMSE = function(Y,Yhat){sqrt(mean((Y - Yhat)^2)) }

(RMSE(db_features.test$overalGradeDiff, pred))

# Predic the Improved for the test data and evaluate the prediction accuracy using confusionMatrix method.
pred=predict(svmC1, newdata=d.test[,fs])
confusionMatrix(pred, d.test$PriceCategory)

dataset <- db_features
x <- db_features.train
y <- db_features.test

# Trying RandomForest
control <- trainControl(method="repeatedcv", number=10, repeats=3)
seed <- 7
metric <- "Accuracy"
set.seed(seed)
mtry <- sqrt(ncol(x))
tunegrid <- expand.grid(.mtry=mtry)
rf_default <- train(x=db_features.train[,fs] , y=db_features.train$overalGradeDiff, method="rf")
print(rf_default)

randomForest =train(x=db_features.train[,fs] , y=db_features.train$overalGradeDiff, method = "rf")

######### CLASSIFICATION MODEL

# Trying an SVM model to classify improved given avgTbwSubs and CountOfVideoandForumEvents

# ----- 1. Split data into train and test set
set.seed(1234)
tr.index= sample(nrow(db_features), nrow(db_features)*0.6)
db_features.train= db_features[tr.index,]
db_features.test = db_features[-tr.index,]
# ---- 2 . Define the feature set
fs = c('avgTbwSubs','countOfVideoandForumEvents')
svmCl = train(x=db_features.train[,fs],y=db_features.train$improved, method = "svmLinear") 
pred=predict(svmCl, newdata=db_features.test[,fs])
RMSE = function(Y,Yhat){sqrt(mean((Y - Yhat)^2)) }


(RMSE(db_features.test$improved, pred))
# Evaluate the prediction accuracy using confusionMatrix method.
confusionMatrix(pred, db_features.test$improved)

