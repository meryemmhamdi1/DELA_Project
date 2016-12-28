library(plyr) #ddply
library(dplyr)
library(AUC) #install.packages('AUC')
library(caret)
library(class)
library(caTools)

db=read.csv('/media/diskD/EPFL/Fall 2016/DELA/DELA_Project/Classification/RawData/TrainAllFeatures.csv', stringsAsFactors = F)

db$IsLastSubm = (ifelse(db$SubmissionsLeft == 0, 1,0))
db$TotalTimeVideo = db$DurationOfVideoActivity*db$DistinctIds
#train_fs = c('Speed','TimeSinceLast','DistinctIds','VideoForumThreads','DurationOfVideoActivity','SelectiveNumOfEvents')

#train_fs = c('SeenVideo','NVideoAndForum_','IsLastSubm','NumberOfThreadsLaunched','SubmissionNumber','TotalTimeVideo','PlaysTimesThreadViews','EngagementIndex')

#'PlaysTimesThreadViews',
train_fs = c('SeenVideo','NVideoAndForum_','IsLastSubm','NumberOfThreadsLaunched','SubmissionNumber','TotalTimeVideo','EngagementIndex')

fs=c("ProblemID",
       "SubmissionNumber",
       "TimeStamp",
       "TimeSinceLast",          
       "NVideoEvents",
       "NForumEvents",
       "NumberOfPlays",          
       "NumberOfPosts",
       "NumberOfComments",
       "DistinctIds",
       "NumberOfLoads",
       "SeenVideo",
       "TotalVideoEvents",
       "TotalForumEvents",
       "NumberOfThreadsLaunched",
       "NumberOfDownloads",
       "ScoreRelevantEvents",    
       "SelectiveNumOfEvents",
       "NumberOfPauses",
       "NumberOfThreadViews",
       "DurationOfVideoActivity",
       "NumberOfSpeedChange",    
       "NVideoAndForum_",
       "EngagementIndex",
       "ComAndPost",
       "AverageVideoTimeDiffs",
       "PlaysDownlsPerVideo",    
       "NVideoAndForum"
)

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

# -- Logistic Regression
ctrl_lr= trainControl(method = 'cv', summaryFunction=twoClassSummary ,classProbs = TRUE)
paramGrid_lr = expand.grid(.nIter = c(1:3))
model_lr = train(x= db.train[,train_fs],
                 y= db.train$Class,
                 method = "LogitBoost",
                 metric="ROC",
                 trControl = ctrl_lr,
                 tuneGrid = paramGrid_lr,
                 preProc = c("center", "scale"))
print(model_lr);   plot(model_lr)  

preds_train_lr = predict(model_lr, newdata=db.train[,train_fs]);
ROC_curve_train_lr = roc(preds_train_lr, db.train$Class); auc(ROC_curve_train_lr)

varImp(model_lr)

#----- check generalizability of your model on new data
preds_test_lr= predict(model_lr, newdata = db.test[,train_fs]);
dim(preds_train_lr)
table(preds_test_lr)

ROC_curve_test_lr = roc(preds_test_lr, db.test$Class);  auc(ROC_curve_test_lr)


library(mlr) # install.packages('mlr')


# Learning Curve for Training and Testing Datasets
# 'PlaysTimesThreadViews'
fs = c('SeenVideo','NVideoAndForum_','IsLastSubm','NumberOfThreadsLaunched','SubmissionNumber','TotalTimeVideo','EngagementIndex','Class')

x=up_train[,fs]
set.seed(29510)
lda_data <- learing_curve_dat(dat = x, 
                              outcome = "Class",
                              test_prop = 0.3, 
                              method = "plr",
                              metric="ROC",
                              trControl = ctrl_rf,
                              tuneGrid = paramGrid_rf,
                              preProc = c("center", "scale"))



ggplot(lda_data, aes(x = Training_Size, y = ROC, color = Data)) + 
  geom_smooth(method = loess, span = .8) + 
  theme_bw()



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
preds_lr = predict(model_lr, newdata =data);


#======================================================================== 
#         step 2.2: prepare submission file for kaggle
#======================================================================== 

cl.Results=testDb[,c('ProblemIDOld', 'UserID', 'SubmissionNumberOld')]
cl.Results$improved= preds_lr #
levels(cl.Results$improved)=c(0,1) # 
cl.Results$uniqRowID= paste0(cl.Results$UserID,'_', cl.Results$ProblemIDOld,'_', cl.Results$SubmissionNumberOld)
cl.Results=cl.Results[,c('uniqRowID','improved')]
table(cl.Results$improved)

#----- keep only rows which are listed in classifier_templtae.csv file
#----- this excludes first submissions and cases with no forum and video event in between two submissions
classifier_template= read.csv('/media/diskD/EPFL/Fall 2016/DELA/DELA_Project/Classification/RawData/classifier_template.csv', stringsAsFactors = F)
kaggleSubmission=merge(classifier_template,cl.Results )
write.csv(kaggleSubmission,file='/media/diskD/EPFL/Fall 2016/DELA/DELA_Project/Classification/Results/classifier_results_lr.csv', row.names = F)


#------- submit the resulting file (classifier_results.csv) to kaggle 
#------- report AUC in private score in your report

