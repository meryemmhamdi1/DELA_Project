library(plyr)
library(dplyr)
library(caret)

#======================================================================== 
#         step 1.1: Prepare Dataset
#======================================================================== 

#------ read features extracted from train set, using your python script
db=read.csv('/home/nevena/Desktop/Digital education/DELA_Project/Regression/RawAndFeatureData/OutputTableTrain.csv', stringsAsFactors = F)

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

<<<<<<< HEAD
db_agg <- ddply(db, .(ProblemID, UserID), summarize, SeenVideoAgg = sum(SeenVideo),NVideoAndForum_Sum = sum(NVideoAndForum_),
                NumberOfThreadsLaunchedSum = sum(NumberOfThreadsLaunched), TotalTimeVideoSum =sum(TotalTimeVideo),
                SubmissionNumberLen = length(SubmissionNumber) , 
                EngagementIndexSum = sum(EngagementIndex), gradeDiff = Grade[SubmissionNumberLen]-Grade[1],
                sumbNumTimesNumVidForumSum = sum(sumbNumTimesNumVidForum),PlaysDownlsPerVideoSum = sum(PlaysDownlsPerVideo),
                SelectiveNumOfEventsSum = sum(SelectiveNumOfEvents), TimeSinceLastSum = sum(TimeSinceLast), IsLastSubm = sum(IsLastSubm) )

# totalDurationVideo = sum(DurationOfVideoActivity), totalDistinctIDs = sum(DistinctIds),
# NumberVideoEvents = sum(TotalVideoEvents), NumberForumEvents = sum(TotalForumEvents), 
=======
db_agg <- ddply(db, .(ProblemID, UserID), summarize, 
                NVideoAndForum_=sum(NVideoAndForum_),
                #NumberOfThreadsLaunchedSum = sum(NumberOfThreadsLaunched),
                SubmissionNumberLen = length(SubmissionNumber),
                #DistIds=sum(DistinctIds),
                #SeenVideoSum=sum(SeenVideo),
                #TotalTimeSum=sum(TotalTimeVideo),
                gradeDiff = Grade[SubmissionNumberLen]-Grade[1] )

>>>>>>> 53b4ab3458d5b1f971962fb30f176f360ad4f3c3
#====================================================================================================== 
#         step 2.1: Applying Regression Algorithms to predict grade difference 
#======================================================================================================

# Visualizing the distribution of grades of students 
hist(db_agg$gradeDiff)
hist(db_agg$avgGrade)
# Upsample the students with better grades since we have many problems/students for which grade didn't improve that much
<<<<<<< HEAD
#train_fs = c("SeenVideoAgg","NVideoAndForum_Sum","TotalTimeVideoSum","NumberOfThreadsLaunchedSum","SubmissionNumberLen","EngagementIndexSum")
=======
fs = c("NVideoAndForum_","SubmissionNumberLen") #,"AverageNForumEventsMean")

## Define categories of grade improvement for upsampling purposes
#db_agg$GradeCategory =  factor(ifelse(db_agg$GradeCategory<-50 ,"Category4", "Other" ))
#db_agg$GradeCategory =  factor(ifelse(-50<=db_agg$GradeCategory<0 ,"Category3", "Other" ))
>>>>>>> 53b4ab3458d5b1f971962fb30f176f360ad4f3c3

train_fs = c("SubmissionNumberLen","TimeSinceLastSum","NVideoAndForum_Sum","SelectiveNumOfEventsSum") #"IsLastSubm","PlaysDownlsPerVideoSum"
# Splitting dataset into train and test without UPSAMPLING

set.seed(1234)
tr.index= sample(nrow(db_agg), nrow(db_agg)*0.7)
db_agg.train= db_agg[tr.index,]
db_agg.test = db_agg[-tr.index,]
dim(db_agg.train)
dim(db_agg.test)


# Splitting dataset into train and test with UPSAMPLING

## Define categories of grade improvement for upsampling purposes
levels <- c(-100, -50,-20, 0, 50, 100)
labels <- c("Category1", "Category2", "Category3", "Category4", "Category5")
db_agg <- db_agg %>% mutate( gradeCategory = cut(db_agg$gradeDiff, levels, labels = labels))

up_train <- upSample(x=db_agg,y= db_agg$gradeCategory)

set.seed(1234)
tr.index= sample(nrow(up_train), nrow(up_train)*0.7)
db_agg.train= up_train[tr.index,]
db_agg.test = up_train[-tr.index,]
dim(db_agg.train)
dim(db_agg.test)


# Trying svmLinear, NNet, BoostLm, Lm, LmStepAIC, Lasso, GBM, LM_noPCA, RF

#======================================================================== 
#         Trying svmLinear
#======================================================================== 
<<<<<<< HEAD

# "totalDistinctIDs","totalDurationVideo",
# ,"NumberVideoEvents","NumberForumEvents"
=======
RMSE = function(Y,Yhat){ sqrt(mean((Y - Yhat)^2)) }
train_fs = c("NVideoAndForum_","SubmissionNumberLen") #,"AverageNForumEventsMean")
>>>>>>> 53b4ab3458d5b1f971962fb30f176f360ad4f3c3
ctrl <- trainControl(## 10-fold CV
  method = "repeatedcv",
  number = 5,
  ## repeated ten times
  repeats = 5,
  classProbs = FALSE)

paramGrid <- expand.grid(C = c(0.001,0.01,0.1,0.5 ,1, 2, 3, 4))
# svmLinear gives 50,73696
<<<<<<< HEAD
svm3=train(x=db_agg.train[,train_fs] , y=db_agg.train$gradeDiff, method = "svmRadial", preProc = c("center", "scale"))
           #, trControl = ctrl,tuneGrid = paramGrid)
=======
svm.tune <- train(x=db_agg.train[,train_fs],
                  y= db_agg.train$gradeDiff,
                  method = "svmRadial",   # Radial kernel
                  tuneLength = 9,					# 9 values of the cost function
                  preProc = c("center","scale"),  # Center and scale data
                  trControl=ctrl,
                  parameters=paramGrid)
svm.tune2 <- train(x=db_agg.train[,train_fs],
                   y= db_agg.train$gradeDiff,
                   method = "svmLinear",
                   preProc = c("center","scale"),
                   trControl=ctrl)	
svm.tune3 <- train(x=db_agg.train[,train_fs],
                   y= db_agg.train$gradeDiff,
                   method = "svmPoly",
                   preProc = c("center","scale"),
                   trControl=ctrl)	
svm.tune4 <- train(x=db_agg.train[,train_fs],
                   y= db_agg.train$gradeDiff,
                   method = "svmRadialCost",
                   preProc = c("center","scale"),
                   trControl=ctrl)	
>>>>>>> 53b4ab3458d5b1f971962fb30f176f360ad4f3c3

# Predict and Calculate RMSE for train dataset
pred_train=predict(svm.tune, newdata=db_agg.train[,train_fs])
(RMSE(db_agg.train$gradeDiff, pred_train))

# Predict and Calculate RMSE for test dataset
pred_test=predict(svm.tune, newdata=db_agg.test[,train_fs])
(RMSE(db_agg.test$gradeDiff, pred_test))

#======================================================================== 
#         Trying LmStepAIC
#======================================================================== 

# lmStepAIC without pre-processing or fine tuning: 47,76091 
cm4 <- train(x=db_agg.train[,train_fs] , y=db_agg.train$gradeDiff, method = "lmStepAIC",trControl = ctrl, preProc = c("center", "scale")) 
#summary(cm4)$r.squared

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
  paramGrid <- expand.grid(n.trees = c(50:100),interaction.depth=c(2,5,10,20,40),shrinkage=c(4,5),n.minobsinnode=c(1,2,3))
  set.seed(2014)
  
  gbm <- train(x=db_agg.train[,train_fs] , y=db_agg.train$gradeDiff, method = "gbm", trControl = ctrl, tuneLength=6)
  
  # Predict and Calculate RMSE for train dataset
  pred_train=predict(gbm, newdata=db_agg.train[,train_fs])
  (RMSE(db_agg.train$gradeDiff, pred_train))
  
  # Predict and Calculate RMSE for test dataset
  pred_test=predict(gbm, newdata=db_agg.test[,train_fs])
  (RMSE(db_agg.test$gradeDiff, pred_test))
#======================================================================== 
#         Trying cart
#========================================================================

<<<<<<< HEAD
# gbm gives 38,90521 even improved with pre-processing to reach 38.67384
paramGrid <- expand.grid(n.trees = c(30:53),interaction.depth=c(2:5),shrinkage=c(0.1),n.minobsinnode=c(9))
control <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid")

gbm <- train(x=db_agg.train[,train_fs] , y=db_agg.train$gradeDiff, method = "gbm",preProc = c("center", "scale"),trControl = control,tuneGrid = paramGrid) 
summary(gbm)
=======
cartModel <- train(x=db_agg.train[,train_fs] , y=db_agg.train$gradeDiff,method = "rpart", trControl = fitControl, tuneLength=5)

pred_train=predict(cartModel, newdata=db_agg.train[,train_fs])
(RMSE(db_agg.train$gradeDiff, pred_train))
>>>>>>> 53b4ab3458d5b1f971962fb30f176f360ad4f3c3

# Predict and Calculate RMSE for test dataset
pred_test=predict(cartModel, newdata=db_agg.test[,train_fs])
(RMSE(db_agg.test$gradeDiff, pred_test))

#======================================================================== 
#         Trying earth
#========================================================================
earthModel <- train(x=db_agg.train[,train_fs] , y=db_agg.train$gradeDiff, method = "earth", trControl = fitControl, tuneLength=18)

pred_train=predict(earthModel, newdata=db_agg.train[,train_fs])
(RMSE(db_agg.train$gradeDiff, pred_train))

# Predict and Calculate RMSE for test dataset
pred_test=predict(earthModel, newdata=db_agg.test[,train_fs])
(RMSE(db_agg.test$gradeDiff, pred_test))
#======================================================================== 
#         Trying Conditional Inference Tree
#========================================================================
eNetModel <- train(x=db_agg.train[,train_fs] , y=db_agg.train$gradeDiff, method = "ctree", trControl = ctrl, tuneLength=8)

pred_train=predict(eNetModel, newdata=db_agg.train[,train_fs])
(RMSE(db_agg.train$gradeDiff, pred_train))

# Predict and Calculate RMSE for test dataset
pred_test=predict(eNetModel, newdata=db_agg.test[,train_fs])
(RMSE(db_agg.test$gradeDiff, pred_test))

#======================================================================== 
#         Trying rf
#========================================================================


# Trying Random Forest
customRF <- list(type = "Regression", library = "randomForest", loop = NULL)
customRF$parameters <- data.frame(parameter = c("mtry", "ntree"), class = rep("numeric", 2), label = c("mtry", "ntree"))
customRF$grid <- function(x, y, len = NULL, search = "grid") {}
customRF$fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
  randomForest(x, y, mtry = param$mtry, ntree=param$ntree, ...)
}
customRF$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata)
customRF$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata, type = "prob")
customRF$sort <- function(x) x[order(x[,1]),]


# rf gives 
paramGrid <- expand.grid(mtry= c(1:9),ntree=c(10:40))

<<<<<<< HEAD
rf <- train(x=db_agg.train[,train_fs] , y=db_agg.train$gradeDiff, method = customRF,trControl = ctrl,tuneGrid = paramGrid, preProc = c("center", "scale")) 
summary(rf)
=======
rf <- train(x=db_agg.train[,train_fs] , y=db_agg.train$gradeDiff, method = "rf",trControl = ctrl,tuneGrid = paramGrid, preProc = c("center", "scale")) 
#summary(rf)
>>>>>>> 53b4ab3458d5b1f971962fb30f176f360ad4f3c3

# Predict and Calculate RMSE for train dataset
pred_train=predict(rf, newdata=db_agg.train[,train_fs])
(RMSE(db_agg.train$gradeDiff, pred_train))

# Predict and Calculate RMSE for test dataset
pred_test=predict(rf, newdata=db_agg.test[,train_fs])
(RMSE(db_agg.test$gradeDiff, pred_test))

#========================================================================
#        Trying NNet
#========================================================================
paramGrid <- expand.grid(size = c(20:100),decay=c(2,5,10,20,30),bag=c(4:15))
  
nnet <- train(x=db_agg.train[,train_fs] , y=db_agg.train$gradeDiff, method = "avNNet",preProc = c("center", "scale"))
#,trControl = ctrl,tuneGrid = paramGrid) 
summary(gbm)$r.squared

# Predict and Calculate RMSE for train dataset
pred_train=predict(gbm, newdata=db_agg.train[,train_fs])
(RMSE(db_agg.train$gradeDiff, pred_train))

# Predict and Calculate RMSE for test dataset
pred_test=predict(gbm, newdata=db_agg.test[,train_fs])
(RMSE(db_agg.test$gradeDiff, pred_test))

#======================================================================== 
#         step 3.1: Use Regressor to predict progress for test data
#======================================================================== 
testDb=read.csv('/home/nevena/Desktop/Digital education/DELA_Project/Regression/RawAndFeatureData/OutputTableTest.csv', stringsAsFactors = F)

# NEW FEATURES 
<<<<<<< HEAD
testDb$TotalTimeVideo = testDb$DurationOfVideoActivity*testDb$DistinctIds
db_test_agg <- ddply(testDb, .(ProblemID, UserID), summarize, SeenVideoAgg = sum(SeenVideo),NVideoAndForum_Sum = sum(NVideoAndForum_),
                     NumberOfThreadsLaunchedSum = sum(NumberOfThreadsLaunched), TotalTimeVideoSum =sum(TotalTimeVideo),
                     SubmissionNumberLen = length(SubmissionNumber) , 
                     EngagementIndexSum = sum(EngagementIndex), gradeDiff = Grade[SubmissionNumberLen]-Grade[1],
                     sumbNumTimesNumVidForumSum = sum(sumbNumTimesNumVidForum),PlaysDownlsPerVideoSum = sum(PlaysDownlsPerVideo),
                     SelectiveNumOfEventsSum = sum(SelectiveNumOfEvents), TimeSinceLastSum = sum(TimeSinceLast), IsLastSubm = sum(IsLastSubm) )
=======

db_test_agg <- ddply(testDb, .(ProblemID, UserID), summarize, 
                     #SeenVideoAgg = sum(SeenVideo),
                     NVideoAndForum_=sum(NVideoAndForum_),
                     SubmissionNumberLen = length(SubmissionNumber),
                     gradeDiff = Grade[SubmissionNumberLen]-Grade[1]  )


>>>>>>> 53b4ab3458d5b1f971962fb30f176f360ad4f3c3

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
regression_template= read.csv('/home/nevena/Desktop/Digital education/DELA_Project/Regression/regression_template.csv', stringsAsFactors = F)
kaggleSubmission=merge(regression_template,cl.Results )
<<<<<<< HEAD
write.csv(kaggleSubmission,file='/media/diskD/EPFL/Fall 2016/DELA/DELA_Project/Regression/Results/regression_results_gbm_new.csv', row.names = F)
=======
write.csv(kaggleSubmission,file='/home/nevena/Desktop/Digital education/DELA_Project/Regression/Results/regression_results_gbm.csv', row.names = F)
>>>>>>> 53b4ab3458d5b1f971962fb30f176f360ad4f3c3

