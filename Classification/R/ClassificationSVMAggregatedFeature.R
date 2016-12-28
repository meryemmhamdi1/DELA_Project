library(plyr) #ddply
library(dplyr)
library(AUC) #install.packages('AUC')
library(caret)
library(class)


library("e1071")
db=read.csv('/media/diskD/EPFL/Fall 2016/DELA/DELA_Project/Classification/RawData/TrainAllFeatures.csv', stringsAsFactors = F)

db$IsLastSubm = (ifelse(db$SubmissionsLeft == 0, 1,0))
db$TotalTimeVideo = db$DurationOfVideoActivity*db$DistinctIds
#train_fs = c('Speed','TimeSinceLast','DistinctIds','VideoForumThreads','DurationOfVideoActivity','SelectiveNumOfEvents')

#train_fs = c('SeenVideo','NVideoAndForum_','IsLastSubm','NumberOfThreadsLaunched','SubmissionNumber','TotalTimeVideo','PlaysTimesThreadViews','EngagementIndex')

# ,'PlaysTimesThreadViews'
train_fs = c('SeenVideo','NVideoAndForum_','IsLastSubm','NumberOfThreadsLaunched','SubmissionNumber','TotalTimeVideo','EngagementIndex')

db$Improved =  factor(ifelse(db$Improved==0 ,"No", "Yes" ))

# UP-SAMPLING
up_train <- upSample(x=db[train_fs],
                     y= db$Improved)

# ----- (Optional) split your training data into train and test set. Use train set to build your classifier and try it on test data to check generalizability. 
set.seed(1234)
tr.index= sample(nrow(up_train), nrow(up_train)*0.7)
db.train= up_train[tr.index,]
db.test = up_train[-tr.index,]
dim(db.train)
dim(db.test)

# PRE-PROCESSING
trainX <- db.train[,train_fs]
preProcValues <- preProcess(x = trainX,method = c("center", "scale"))

# TRYING SVM Model
svm.model <- svm(db.train$Class ~., data = db.train[,train_fs], kernel  =
                   "radial", cost = 100, degree = 3, gamma = 2, coef.0 = 0,
                 epsilon = 0.1)

# Training error
pred_svm_train <- predict(svm.model, db.train[,train_fs])
auc(roc(pred_svm_train, db.train$Class))

table(db.train$Class, pred_svm_train)

# Testing error
pred_svm_test <- predict(svm.model, db.test[,train_fs])
auc(roc(pred_svm_test, db.test$Class))

table(db.test$Class, pred_svm_test)


# Tuning SVM : TRY WITH DIFFERENT KERNELS: RADIAL, LINEAR, POLYNOMIAL, ...


svm_tune <- tune(svm, train.x=db.train[,train_fs], train.y=db.train$Class, 
                 kernel="radial", ranges=list(cost=10^(-1:2),  gamma=c(.5,1,2)))

print(svm_tune)
summary(svm_tune)

# Reapplying SVM after Tuning:

svm_model_after_tune <- svm(db.train$Class ~ ., data=db.train[,train_fs], kernel="radial", cost=0.1, gamma=1)
summary(svm_model_after_tune)

# Run the model to get training and testing predictions: 

# Training error
pred_svm_tune_train <- predict(svm_model_after_tune, db.train[,train_fs])
auc(roc(pred_svm_tune_train, db.train$Class))

table(db.train$Class, pred_svm_tune_train)

# Testing error
pred_svm_tune_test <- predict(svm_model_after_tune, db.test[,train_fs])
auc(roc(pred_svm_tune_test, db.test$Class))

table(db.test$Class, pred_svm_tune_test)

# Learning Curve for Training and Testing Datasets




#======================================================================== 
#         step 2.1: Use classifier to predict progress for test data
#======================================================================== 

testDb=read.csv('/media/diskD/EPFL/Fall 2016/DELA/DELA_Project/Classification/RawData/TestAllFeatures.csv', stringsAsFactors = F)
testDb$ProblemIDOld = testDb$ProblemID
testDb$SubmissionNumberOld = testDb$SubmissionNumber

# NEW FEATURES 
testDb$IsLastSubm = (ifelse(testDb$SubmissionsLeft == 0, 1,0))
testDb$TotalTimeVideo = testDb$DurationOfVideoActivity*testDb$DistinctIds

testDb$Grade=NULL; testDb$GradeDiff=NULL;

testDb[is.na(testDb)]=0
data = testDb[,train_fs];
#---- use trained model to predict progress for test data
preds_svm  = predict(svm_model_after_tune, newdata =data);


#======================================================================== 
#         step 2.2: prepare submission file for kaggle
#======================================================================== 

cl.Results=testDb[,c('ProblemIDOld', 'UserID', 'SubmissionNumberOld')]
cl.Results$improved= preds_svm #
levels(cl.Results$improved)=c(0,1) # 
cl.Results$uniqRowID= paste0(cl.Results$UserID,'_', cl.Results$ProblemIDOld,'_', cl.Results$SubmissionNumberOld)
cl.Results=cl.Results[,c('uniqRowID','improved')]
table(cl.Results$improved)

#----- keep only rows which are listed in classifier_templtae.csv file
#----- this excludes first submissions and cases with no forum and video event in between two submissions
classifier_template= read.csv('/media/diskD/EPFL/Fall 2016/DELA/DELA_Project/Classification/RawData/classifier_template.csv', stringsAsFactors = F)
kaggleSubmission=merge(classifier_template,cl.Results )
write.csv(kaggleSubmission,file='/media/diskD/EPFL/Fall 2016/DELA/DELA_Project/Classification/Results/classifier_results_svm.csv', row.names = F)


#------- submit the resulting file (classifier_results.csv) to kaggle 
#------- report AUC in private score in your report

