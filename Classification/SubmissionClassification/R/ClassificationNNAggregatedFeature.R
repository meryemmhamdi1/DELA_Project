library(plyr) #ddply
library(dplyr)
library(AUC) #install.packages('AUC')
library(caret)
library(class)

db=read.csv('/media/diskD/EPFL/Fall 2016/DELA/DELA_Project/Classification/RawData/TrainAllFeatures.csv', stringsAsFactors = F)

db$IsLastSubm = (ifelse(db$SubmissionsLeft == 0, 1,0))
db$TotalTimeVideo = db$DurationOfVideoActivity*db$DistinctIds
#train_fs = c('Speed','TimeSinceLast','DistinctIds','VideoForumThreads','DurationOfVideoActivity','SelectiveNumOfEvents')

#train_fs = c('SeenVideo','NVideoAndForum_','IsLastSubm','NumberOfThreadsLaunched','SubmissionNumber','TotalTimeVideo','PlaysTimesThreadViews','EngagementIndex')

train_fs = c('SeenVideo','NVideoAndForum_','IsLastSubm','NumberOfThreadsLaunched','SubmissionNumber','TotalTimeVideo','PlaysTimesThreadViews','EngagementIndex')

db$Improved =  factor(ifelse(db$Improved==0 ,"No", "Yes" ))

up_train <- upSample(x=db[train_fs],
                     y= db$Improved)

# ----- (Optional) split your training data into train and test set. Use train set to build your classifier and try it on test data to check generalizability. 
set.seed(1234)
tr.index= sample(nrow(up_train), nrow(up_train)*0.7)
db.train= up_train[tr.index,]
db.test = up_train[-tr.index,]
dim(db.train)
dim(db.test)

# Trying K Nearest Neighbors

ctrl <- trainControl(method = 'cv', number = 10, summaryFunction=twoClassSummary ,classProbs = TRUE)

paramGrid <- expand.grid(k = range(1,10))

knnFit=train(x=up_train[,train_fs],
             y=up_train$Class,
             method = "knn",
             metric="ROC",
             trControl = ctrl,
             tuneGrid = paramGrid,
             preProc = c("center", "scale"))

plot(knnFit)

preds_train_knn= predict(knnFit, newdata = up_train[,train_fs]);
table(preds_train_knn)

auc(roc(preds_train_knn, up_train$Class));


preds_test_knn= predict(knnFit, newdata = db.test[,train_fs]);
table(preds_test_knn)

auc(roc(preds_test_knn, db.test$Class))


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
preds_knn= predict(knnFit, newdata = data);


#======================================================================== 
#         step 2.2: prepare submission file for kaggle
#======================================================================== 

cl.Results=testDb[,c('ProblemIDOld', 'UserID', 'SubmissionNumberOld')]
cl.Results$improved= preds_knn #
levels(cl.Results$improved)=c(0,1) # 
cl.Results$uniqRowID= paste0(cl.Results$UserID,'_', cl.Results$ProblemIDOld,'_', cl.Results$SubmissionNumberOld)
cl.Results=cl.Results[,c('uniqRowID','improved')]
table(cl.Results$improved)

#----- keep only rows which are listed in classifier_templtae.csv file
#----- this excludes first submissions and cases with no forum and video event in between two submissions
classifier_template= read.csv('/media/diskD/EPFL/Fall 2016/DELA/DELA_Project/Classification/RawData/classifier_template.csv', stringsAsFactors = F)
kaggleSubmission=merge(classifier_template,cl.Results )
write.csv(kaggleSubmission,file='/media/diskD/EPFL/Fall 2016/DELA/DELA_Project/Classification/Results/classifier_results_knn.csv', row.names = F)


#------- submit the resulting file (classifier_results.csv) to kaggle 
#------- report AUC in private score in your report

