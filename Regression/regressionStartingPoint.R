library(plyr)
library(dplyr)
library(caret)

#======================================================================== 
#         step 1.1: Prepare Dataset
#======================================================================== 

#------ read features extracted from train set, using your python script
db=read.csv('/media/diskD/EPFL/Fall 2016/DELA/DELA_Project/Regression/RawAndFeatureData/OutputTableTrain.csv', stringsAsFactors = F)

#------ sort submissions
db=db[order(db$UserID,db$ProblemID,db$SubmissionNumber),]
db$TotalTimeVideo = db$DurationOfVideoActivity*db$DistinctIds

#--- replace NA values with 0
db[is.na(db)]=0

#----- remove first submissions
#db = db[db$SubmissionNumber>0,]
db= filter(db,SubmissionNumber>0)

#---- remove cases when there is no video or forum activity between two submissions
db$NVideoAndForum= db$NVideoEvents+db$NForumEvents
db= filter(db,NVideoAndForum>0)  

#====================================================================================================== 
#         step 1.2: Aggregate Features for each student and problem between First and Last Submission
#======================================================================================================
# count <- ddply(db, .(ProblemID, UserID), summarize, freq = length(ProblemID))

# Example Dataframe in r
# problemID <- c('Problem1','Problem1','Problem2','Problem2','Problem2','Problem1','Problem1','Problem1','Problem2','Problem2','Problem3','Problem3')
# userID <- c('user1','user1','user1','user1','user1','user2','user2','user2','user2','user2','user3','user3')
# submissionNumber <- c(1,2,1,2,3,1,2,3,1,2,1,2)
# grades <- c(50,90,100,90,70,50,60,80,20,30,10,40)
# submissionFrame <- data.frame(problemID,userID,submissionNumber,grades)
# submissionNumberagg <- ddply(submissionFrame, .(problemID,userID), summarize, V2 = sum(submissionNumber), V3 = length(submissionNumber),gradeDiff = grades[V3]-grades[1])

db_agg <- ddply(db, .(ProblemID, UserID), summarize, SeenVideoAgg = sum(SeenVideo),
                NVideoAndForum_Sum = sum(NVideoAndForum_),NumberOfThreadsLaunchedSum = sum(NumberOfThreadsLaunched),
                SubmissionNumberLen = length(SubmissionNumber), TotalTimeVideoSum =sum(TotalTimeVideo),
                EngagementIndexSum = sum(EngagementIndex), gradeDiff = Grade[SubmissionNumberLen]-Grade[1]  )

#====================================================================================================== 
#         step 2.1: Applying Regression Algorithms to predict grade difference 
#======================================================================================================

# Visualizing the distribution of grades of students 
hist(db_agg$gradeDiff)

# Upsample the students with better grades since we have many problems/students for which grade didn't improve that much
fs = c("SeenVideoAgg","NVideoAndForum_Sum","NumberOfThreadsLaunchedSum","SubmissionNumberLen","TotalTimeVideoSum","EngagementIndexSum","gradeDiff")

## Define categories of grade improvement for upsampling purposes
#db_agg$GradeCategory =  factor(ifelse(db_agg$GradeCategory<-50 ,"Category4", "Other" ))
#db_agg$GradeCategory =  factor(ifelse(-50<=db_agg$GradeCategory<0 ,"Category3", "Other" ))

#up_train <- upSample(x=db_agg[train_fs],y= db_agg$gradeDiff)

# Splitting dataset into train and test 
set.seed(1234)
tr.index= sample(nrow(db_agg), nrow(db_agg)*0.7)
db_agg.train= db_agg[tr.index,]
db_agg.test = db_agg[-tr.index,]
dim(db_agg.train)
dim(db_agg.test)

# Trying svmLinear, NNet, BoostLm, Lm, LmStepAIC, Lasso, GBM, LM_noPCA, RF

#======================================================================== 
#         Trying svmLinear
#======================================================================== 

train_fs = c("SeenVideoAgg","NVideoAndForum_Sum","NumberOfThreadsLaunchedSum","SubmissionNumberLen","TotalTimeVideoSum","EngagementIndexSum") #,"AverageNForumEventsMean")
ctrl <- trainControl(## 10-fold CV
  method = "repeatedcv",
  number = 10,
  ## repeated ten times
  repeats = 10,
  classProbs = FALSE)
paramGrid <- expand.grid(C = c(0.001,0.01,0.1,0.5 ,1, 2, 3, 4))
# svmLinear gives 50,73696
svm3=train(x=db_agg.train[,train_fs] , y=db_agg.train$gradeDiff, method = "svmLinear"
           , trControl = ctrl,tuneGrid = paramGrid, preProc = c("center", "scale"))

# Predict and Calculate RMSE for train dataset
pred_train=predict(svm3, newdata=db_agg.train[,train_fs])
(RMSE(db_agg.train$gradeDiff, pred_train))

# Predict and Calculate RMSE for test dataset
pred_test=predict(svm3, newdata=db_agg.test[,train_fs])
(RMSE(db_agg.test$gradeDiff, pred_test))

#======================================================================== 
#         Trying LmStepAIC
#======================================================================== 

# lmStepAIC without pre-processing or fine tuning: 47,76091 
cm4 <- train(x=db_agg.train[,train_fs] , y=db_agg.train$gradeDiff, method = "lmStepAIC",trControl = ctrl, preProc = c("center", "scale")) 
summary(cm4)$r.squared

# Define RMSE
RMSE = function(Y,Yhat){ sqrt(mean((Y - Yhat)^2)) }

# Predict and Calculate RMSE for train dataset
pred_train=predict(cm4, newdata=db_agg.train[,train_fs])
(RMSE(db_agg.train$gradeDiff, pred_train))

# Predict and Calculate RMSE for test dataset
pred_test=predict(cm4, newdata=db_agg.test[,train_fs])
(RMSE(db_agg.test$gradeDiff, pred_test))

#======================================================================== 
#         Trying gbm
#========================================================================

# gbm gives 38,90521 even improved with pre-processing to reach 38.67384
paramGrid <- expand.grid(n.trees = c(50:100),interaction.depth=c(2,5,10,20),shrinkage=c(4,5),n.minobsinnode=c(1,2,3))

gbm <- train(x=db_agg.train[,train_fs] , y=db_agg.train$gradeDiff, method = "gbm",preProc = c("center", "scale"))
             #,trControl = ctrl,tuneGrid = paramGrid) 
summary(gbm)$r.squared

# Predict and Calculate RMSE for train dataset
pred_train=predict(gbm, newdata=db_agg.train[,train_fs])
(RMSE(db_agg.train$gradeDiff, pred_train))

# Predict and Calculate RMSE for test dataset
pred_test=predict(gbm, newdata=db_agg.test[,train_fs])
(RMSE(db_agg.test$gradeDiff, pred_test))

#======================================================================== 
#         Trying rf
#========================================================================

# rf gives 
paramGrid <- expand.grid(mtry= c(1:9))

rf <- train(x=db_agg.train[,train_fs] , y=db_agg.train$gradeDiff, method = "rf",trControl = ctrl,tuneGrid = paramGrid, preProc = c("center", "scale")) 
summary(rf)

# Predict and Calculate RMSE for train dataset
pred_train=predict(rf, newdata=db_agg.train[,train_fs])
(RMSE(db_agg.train$gradeDiff, pred_train))

# Predict and Calculate RMSE for test dataset
pred_test=predict(rf, newdata=db_agg.test[,train_fs])
(RMSE(db_agg.test$gradeDiff, pred_test))


#======================================================================== 
#         step 3.1: Use Regressor to predict progress for test data
#======================================================================== 
testDb=read.csv('/media/diskD/EPFL/Fall 2016/DELA/DELA_Project/Regression/RawAndFeatureData/OutputTableTest.csv', stringsAsFactors = F)

# NEW FEATURES 
testDb$TotalTimeVideo = testDb$DurationOfVideoActivity*testDb$DistinctIds
db_test_agg <- ddply(testDb, .(ProblemID, UserID), summarize, SeenVideoAgg = sum(SeenVideo),
                NVideoAndForum_Sum = sum(NVideoAndForum_),NumberOfThreadsLaunchedSum = sum(NumberOfThreadsLaunched),
                SubmissionNumberLen = length(SubmissionNumber), TotalTimeVideoSum =sum(TotalTimeVideo),
                EngagementIndexSum = sum(EngagementIndex))


testDb$Grade=NULL; testDb$GradeDiff=NULL;

db_test_agg[is.na(db_test_agg)]=0

#---- use trained model to predict progress for test data
pred_lm=predict(rf, newdata=db_test_agg[,train_fs])


#======================================================================== 
#         step 3.2: prepare submission file for kaggle
#======================================================================== 

cl.Results=db_test_agg[,c('ProblemID', 'UserID')]
cl.Results$overalGradeDiff= pred_lm 
cl.Results$uniqRowID= paste0(cl.Results$UserID,'_', cl.Results$ProblemID)
cl.Results=cl.Results[,c('uniqRowID','overalGradeDiff')]

#----- keep only rows which are listed in regression_template.csv file
#----- this excludes first submissions and cases with no forum and video event in between two submissions
regression_template= read.csv('/media/diskD/EPFL/Fall 2016/DELA/DELA_Project/Regression/regression_template.csv', stringsAsFactors = F)
kaggleSubmission=merge(regression_template,cl.Results )
write.csv(kaggleSubmission,file='/media/diskD/EPFL/Fall 2016/DELA/DELA_Project/Regression/Results/regression_results_rf.csv', row.names = F)

