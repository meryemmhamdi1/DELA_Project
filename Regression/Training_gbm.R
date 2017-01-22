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

db_agg <- ddply(db, .(ProblemID, UserID), summarize, SeenVideoAgg = sum(SeenVideo),NVideoAndForum_Sum = sum(NVideoAndForum_),
                NumberOfThreadsLaunchedSum = sum(NumberOfThreadsLaunched), TotalTimeVideoSum =sum(TotalTimeVideo),
                SubmissionNumberLen = length(SubmissionNumber) , 
                EngagementIndexSum = sum(EngagementIndex), gradeDiff = Grade[SubmissionNumberLen]-Grade[1],
                sumbNumTimesNumVidForumSum = sum(sumbNumTimesNumVidForum),PlaysDownlsPerVideoSum = sum(PlaysDownlsPerVideo),
                SelectiveNumOfEventsSum = sum(SelectiveNumOfEvents), TimeSinceLastSum = sum(TimeSinceLast), IsLastSubm = sum(IsLastSubm) )

train_fs = c("SubmissionNumberLen","TimeSinceLastSum","NVideoAndForum_Sum","IsLastSubm") #"SelectiveNumOfEventsSum","PlaysDownlsPerVideoSum"


# Splitting dataset into train and test without UPSAMPLING

set.seed(1234)
tr.index= sample(nrow(db_agg), nrow(db_agg)*0.7)
db_agg.train= db_agg[tr.index,]
db_agg.test = db_agg[-tr.index,]
dim(db_agg.train)
dim(db_agg.test)


#======================================================================== 
#         Training gbm
#========================================================================
# gbm gives 38,90521 even improved with pre-processing to reach 38.67384
paramGrid <- expand.grid(n.trees = c(30:53),interaction.depth=c(2:5),shrinkage=c(0.1),n.minobsinnode=c(7:12))
control <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid")

gbm <- train(x=db_agg.train[,train_fs] , y=db_agg.train$gradeDiff, method = "gbm",preProc = c("center", "scale"),trControl = control,tuneGrid = paramGrid) 
summary(gbm)

# Predict and Calculate RMSE for train dataset
pred_train=predict(gbm, newdata=db_agg.train[,train_fs])
(RMSE(db_agg.train$gradeDiff, pred_train))

# Predict and Calculate RMSE for test dataset
pred_test=predict(gbm, newdata=db_agg.test[,train_fs])
(RMSE(db_agg.test$gradeDiff, pred_test))


#======================================================================== 
#         step 3.1: Use Regressor to predict progress for test data
#======================================================================== 
testDb=read.csv('/media/diskD/EPFL/Fall 2016/DELA/DELA_Project/Regression/RawAndFeatureData/OutputTableTest.csv', stringsAsFactors = F)

# NEW FEATURES 
testDb$TotalTimeVideo = testDb$DurationOfVideoActivity*testDb$DistinctIds
db_test_agg <- ddply(testDb, .(ProblemID, UserID), summarize, SeenVideoAgg = sum(SeenVideo),NVideoAndForum_Sum = sum(NVideoAndForum_),
                     NumberOfThreadsLaunchedSum = sum(NumberOfThreadsLaunched), TotalTimeVideoSum =sum(TotalTimeVideo),
                     SubmissionNumberLen = length(SubmissionNumber) , 
                     EngagementIndexSum = sum(EngagementIndex), gradeDiff = Grade[SubmissionNumberLen]-Grade[1],
                     sumbNumTimesNumVidForumSum = sum(sumbNumTimesNumVidForum),PlaysDownlsPerVideoSum = sum(PlaysDownlsPerVideo),
                     SelectiveNumOfEventsSum = sum(SelectiveNumOfEvents), TimeSinceLastSum = sum(TimeSinceLast), IsLastSubm = sum(IsLastSubm) )

testDb$Grade=NULL; testDb$GradeDiff=NULL;

db_test_agg[is.na(db_test_agg)]=0

#---- use trained model to predict progress for test data
pred_lm=predict(gbm, newdata=db_test_agg[,train_fs])


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
write.csv(kaggleSubmission,file='/media/diskD/EPFL/Fall 2016/DELA/DELA_Project/Regression/Results/regression_results_gbm_new.csv', row.names = F)

