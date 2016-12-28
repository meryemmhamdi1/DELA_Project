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

# Trying Random Forest
customRF <- list(type = "Classification", library = "randomForest", loop = NULL)
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
customRF$levels <- function(x) x$classes

# -- Random Forest Model
ctrl_rf= trainControl(method = 'cv', summaryFunction=twoClassSummary ,classProbs = TRUE)
paramGrid_rf = expand.grid(.mtry = c(1:8) ,.ntree = c(10,20,30,40,50))
model_rf = train(x= up_train[,train_fs],
                 y=up_train$Class,
                 method = customRF,
                 metric="ROC",
                 trControl = ctrl_rf,
                 tuneGrid = paramGrid_rf,
                 preProc = c("center", "scale"))
print(model_rf);   plot(model_rf)  

preds_train_rf = predict(model_rf, newdata=db.train[,train_fs]);
ROC_curve_train_rf = roc(preds_train_rf, db.train$Class); auc(ROC_curve_train_rf)

varImp(model_rf)

#----- check generalizability of your model on new data
preds_test_rf= predict(model_rf, newdata = db.test[,train_fs]);
dim(preds_train_rf)
table(preds_test_rf)

ROC_curve_test_rf = roc(preds_test_rf, db.test$Class);  auc(ROC_curve_test_rf)

install.packages('mlr')
library(mlr)


# Learning Curve for Training and Testing Datasets
r = generateLearningCurveData(
  learners = list("classif.rpart", "classif.knn"),
  task = sonar.task,
  percs = seq(0.1, 1, by = 0.2),
  measures = list(tp, fp, tn, fn),
  resampling = makeResampleDesc(method = "CV", iters = 5),
  show.info = FALSE)


r <- generateLearningCurveData(
  "regr.glmnet",
   bh.task,
   makeResampleDesc(method = "cv", iters = 5, predict = "both"),
   seq(0.1, 1, by = 0.1),
   list(setAggregation(auc, train.mean), setAggregation(auc, test.mean))
)
plotLearningCurve(r)



require(caret)

## Not run: 
set.seed(1412)
class_dat <- twoClassSim(1000)

fs = c('SeenVideo','NVideoAndForum_','IsLastSubm','NumberOfThreadsLaunched','SubmissionNumber','TotalTimeVideo','PlaysTimesThreadViews','EngagementIndex','Class')

x=up_train[,fs]
set.seed(29510)
lda_data <- learing_curve_dat(dat = x, 
                              outcome = "Class",
                              test_prop = 0.3, 
                              ## `train` arguments:
                              method = customRF,
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
preds_rf = predict(model_rf, newdata =data);


#======================================================================== 
#         step 2.2: prepare submission file for kaggle
#======================================================================== 

cl.Results=testDb[,c('ProblemIDOld', 'UserID', 'SubmissionNumberOld')]
cl.Results$improved= preds_rf #
levels(cl.Results$improved)=c(0,1) # 
cl.Results$uniqRowID= paste0(cl.Results$UserID,'_', cl.Results$ProblemIDOld,'_', cl.Results$SubmissionNumberOld)
cl.Results=cl.Results[,c('uniqRowID','improved')]
table(cl.Results$improved)

#----- keep only rows which are listed in classifier_templtae.csv file
#----- this excludes first submissions and cases with no forum and video event in between two submissions
classifier_template= read.csv('/media/diskD/EPFL/Fall 2016/DELA/DELA_Project/Classification/RawData/classifier_template.csv', stringsAsFactors = F)
kaggleSubmission=merge(classifier_template,cl.Results )
write.csv(kaggleSubmission,file='/media/diskD/EPFL/Fall 2016/DELA/DELA_Project/Classification/Results/classifier_results_last_improved.csv', row.names = F)


#------- submit the resulting file (classifier_results.csv) to kaggle 
#------- report AUC in private score in your report

