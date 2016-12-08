library(plyr) #ddply
library(dplyr)
library(AUC) #install.packages('AUC')
library(caret)
library("e1071") #install.packages('e1071')

#======================================================================== 
#         step 1.1: train classifier
#======================================================================== 

#------ read features extracted from train set, using your python script
db=read.csv('/media/diskD/EPFL/Fall 2016/DELA/DELA_Project/Classification/OutputTableTrain.csv', stringsAsFactors = F)

#------ sort submissions
db=db[order(db$UserID,db$ProblemID,db$SubmissionNumber),]

#--- replace NA values with 0
db[is.na(db)]=0

#----- remove first submissions
db= filter(db,SubmissionNumber>0)

#---- remove cases when there is no video or forum activity between two submissions
db$NVideoAndForum= db$NVideoEvents+db$NForumEvents
db= filter(db,NVideoAndForum>0)  

#----- make a catgorical vribale, indicating if grade improved
db$improved = factor(ifelse(db$GradeDiff>0 ,'Yes', 'No' ))
table(db$improved)

# ------------ 
# ----- Whole Feature Set:
# 'ProblemID','UserID','SubmissionNumber','TimeStamp','TimeSinceLast'
# 'NVideoEvents','NForumEvents','NumberOfPlays','NumberOfPosts'
# 'NumberOfComments','SeenVideo','NumberOfDownloads','NumberOfPauses','NumberOfThreadViews'
# 'DurationOfVideoActivity','ScoreRelevantEvents','AverageVideoTimeDiffs'

fs=c('TimeSinceLast','NVideoEvents','NForumEvents','NumberOfPlays','NumberOfPosts','NumberOfComments','SeenVideo','NumberOfDownloads','NumberOfPauses','NumberOfThreadViews','DurationOfVideoActivity','ScoreRelevantEvents','AverageVideoTimeDiffs')

# Normalization of features:
scalar1 <- function(x) {(x-min(x))/(max(x)-min(x))}
for (feature in fs){
  db[feature] = scalar1(db[feature])
}


# ----- (Optional) split your training data into train and test set. Use train set to build your classifier and try it on test data to check generalizability. 
set.seed(1234)
tr.index= sample(nrow(db), nrow(db)*0.9)
db.train= db[tr.index,]
db.test = db[-tr.index,]
dim(db.train)
dim(db.test)


# TRYING SVM Model: 

svm_model <- svm(db.train$improved ~ ., data=db.train[,fs])
summary(svm_model)


# Making predictions:
preds_train_svm = predict(svm_model, newdata=db.train);

ROC_curve_train_svm = roc(preds_train_svm, db.train$improved);
auc(ROC_curve_train_svm)

#----- check generalizability of your model on new data
preds_test_svm = predict(svm_model, newdata=db.test);
table(preds_test_svm)
ROC_curve_test_svm= roc(preds_test_svm, db.test$improved);  
auc(ROC_curve_test_svm)

svm_tune <- tune(svm, train.x=db.train[,fs[1:6]], train.y=db.train$improved, 
                 kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(.5,1,2)))

print(svm_tune)


# Parameter tuning of ‘svm’:
# - sampling method: 10-fold cross validation 
# - best parameters:
# cost gamma
# 0.1   0.5

#- best performance: 0.3435784

# Running SVM with the best parameters found during tuning:

for (feature in features){
  svm_model <- svm(db.train$improved ~ ., data=db.train[,fs], kernel="kernel",cost=0.1,gamma=0.5)
}



#----- train classifier to predict 'improved' status 
#----- Try different methods, model parameters, feature sets and find the best classifier 
#----- Use AUC as model evaluation metric
# ---- PARAMETERS FOR SVM



#### -- SVM Linear Model Using Caret
ctrl_svm = trainControl(method='repeatedcv', repeats=10, number=10, returnResamp = 'none', returnData= FALSE, allowParallel=TRUE, classProbs=TRUE)
model_svm =train(x=db.train[,fs],
                 y=db.train$improved,
                 method = "svmLinear",
                 trControl = ctrl_svm,
                 metric="ROC",
                 tuneGrid = expand.grid(.C=3^(-15:15)),   
                 preProcess = c('center', 'scale'))
print(model_svm);   
plot(model_svm)  

preds_train_svm = predict(model_svm, newdata=db.train);

ROC_curve_train_svm = roc(preds_train_svm, db.train$improved);
auc(ROC_curve_train_svm)

#----- check generalizability of your model on new data
preds_test_svm = predict(model_svm, newdata=db.test);
table(preds_svm)
ROC_curve_test_svm= roc(preds_test_svm, db.test$improved);  
auc(ROC_curve_test_svm)

#======================================================================== 
#         step 1.2: Plot Confusion Matrix
#======================================================================== 
table(preds_train_svm,db.train$improved)
table(preds_test_svm,db.test$improved)

#======================================================================== 
#         step 2.1: Use classifier to predict progress for test data
#======================================================================== 

testDb=read.csv('/media/diskD/EPFL/Fall 2016/DELA/DELA_Project/Classification/OutputTableTest.csv', stringsAsFactors = F)
testDb$Grade=NULL; testDb$GradeDiff=NULL;
testDb[is.na(testDb)]=0

#---- use trained model to predict progress for test data
preds_svm = predict(svm_model, newdata=testDb);

#======================================================================== 
#         step 2.2: prepare submission file for kaggle
#======================================================================== 

cl.Results=testDb[,c('ProblemID', 'UserID', 'SubmissionNumber')]
cl.Results$improved=preds_svm
levels(cl.Results$improved)=c(0,1) # 
cl.Results$uniqRowID= paste0(cl.Results$UserID,'_', cl.Results$ProblemID,'_', cl.Results$SubmissionNumber)
cl.Results=cl.Results[,c('uniqRowID','improved')]
table(cl.Results$improved)

#----- keep only rows which are listed in classifier_templtae.csv file
#----- this excludes first submissions and cases with no forum and video event in between two submissions
classifier_template= read.csv('/media/diskD/EPFL/Fall 2016/DELA/DELA_Project/Classification/classifier_template.csv', stringsAsFactors = F)
kaggleSubmission=merge(classifier_template,cl.Results )
write.csv(kaggleSubmission,file='/media/diskD/EPFL/Fall 2016/DELA/DELA_Project/Classification/classifier_results.csv', row.names = F)


#------- submit the resulting file (classifier_results.csv) to kaggle 
#------- report AUC in private score in your report


