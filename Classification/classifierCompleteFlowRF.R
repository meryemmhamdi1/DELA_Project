library(plyr) #ddply
library(dplyr)
library(AUC) #install.packages('AUC')
library(caret)

#======================================================================== 
#         step 1.1: train classifier
#======================================================================== 

  #------ read features extracted from train set, using your python script
  db=read.csv('/home/nevena/Desktop/Digital education/DELA_Project/Classification/OutputTableTrain.csv', stringsAsFactors = F)
  
  #------ sort submissions
  db=db[order(db$UserID,db$ProblemID,db$SubmissionNumber),]
  
  #--- replace NA values with 0
  db[is.na(db)]=0
  
  #----- remove first submissions
  #db = db[db$SubmissionNumber>0,]
  db= filter(db,SubmissionNumber>0)
  
  #---- remove cases when there is no video or forum activity between two submissions
  db$NVideoAndForum= db$NVideoEvents+db$NForumEvents
  # db = db[db$NVideoAndForum>0,]
  db= filter(db,NVideoAndForum>0)  
  
  #----- make a categorical variable, indicating if grade improved
  db$improved = factor(ifelse(db$GradeDiff>0 ,"Yes", "No" ))
  table(db$improved)
  
  # ----- (Optional) split your training data into train and test set. Use train set to build your classifier and try it on test data to check generalizability. 
  set.seed(1234)
  tr.index= sample(nrow(db), nrow(db)*0.9)
  db.train= db[tr.index,]
  db.test = db[-tr.index,]
  dim(db.train)
  dim(db.test)
  
  #======================================================================== 
  #         Print correlation
  #========================================================================   
  
  #correlationMatrix <- cor(db.train[,5:28])
  # summarize the correlation matrix
  #print(correlationMatrix)
  # find attributes that are highly corrected (ideally >0.75)
  #highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.5)
  # print indexes of highly correlated attributes
  #print(highlyCorrelated)
  
  
  #----- train classifier to predict 'improved' status 
  #----- Try different methods, model parameters, feature sets and find the best classifier 
  #----- Use AUC as model evaluation metric
  # ---- PARAMETERS FOR RANDOM FOREST
  # ------------ ntree : number of trees to grow
  # ------------ mtry: number of variables randomly sampled as candidates at each split
  # ------------ 
  # ----- Whole Feature Set:
  # 'ProblemID','UserID','SubmissionNumber','TimeStamp','TimeSinceLast'
  # 'NVideoEvents','NForumEvents','NumberOfPlays','NumberOfPosts'
  # 'NumberOfComments','SeenVideo','NumberOfDownloads','NumberOfPauses','NumberOfThreadViews'
  # 'DurationOfVideoActivity','ScoreRelevantEvents','AverageVideoTimeDiffs'
  
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
  
  paramGrid_rf <- expand.grid(.mtry = c(5,6,7,8,9) ,.ntree = c(10,20,30,40,50,60,70))

  fs=c(1,2,3,4,5,8:28)
  important_fs = c('NVideoEvents','NumberOfPlays','NumberOfPosts','NumberOfComments','SeenVideo','NumberOfDownloads','NumberOfPauses','DurationOfVideoActivity','AverageVideoTimeDiffs','DistinctIds', 'PlaysDownlsPerVideo','ComAndPost','NumberOfThreadsLaunched')
  ctrl_rf= trainControl(method = 'cv', summaryFunction=twoClassSummary ,classProbs = TRUE)
  # -- Random Forest Model
  model_rf =train(x=db.train[,fs],
               y=db.train$improved,
               method = customRF,
               metric="ROC",
               trControl = ctrl_rf,
               tuneGrid = paramGrid_rf,
               preProc = c("center", "scale"))
  print(model_rf);   plot(model_rf)  
  
  preds_train_rf = predict(model_rf, newdata=db.train);
  ROC_curve_train_rf = roc(preds_train_rf, db.train$improved); auc(ROC_curve_train_rf)
  
#----- check generalizability of your model on new data
  preds_test_rf= predict(model_rf, newdata=db.test);
  dim(preds_train_rf)
  table(preds_test_rf)
  
  ROC_curve_test_rf = roc(preds_test_rf, db.test$improved);  auc(ROC_curve_test_rf)
  
#======================================================================== 
#         step 1.2: Plot Confusion Matrix
#======================================================================== 
  table(preds_train_rf,db.train$improved)
  table(preds_test_rf,db.test$improved)
  
#======================================================================== 
#         step 1.3: Plot Feature/Variable Importance
#======================================================================== 
  
#  Importance
#  AverageVideoTimeDiffs     100.0000
#  DurationOfVideoActivity    98.9172
#  NVideoEvents               98.2282
#  DistinctIds                97.4502
#  SeenVideo                  95.0602
#  PlaysDownlsPerVideo        93.5078
#  NumberOfThreadsLaunched    88.8814
#  NumberOfPlays              87.1383
#  NumberOfPauses             78.0916
#  NumberOfComments           69.2994
#  NumberOfDownloads          69.1539
#  ComAndPost                 63.9989
#  NumberOfPosts              63.7779
#  NForumEvents                4.9693
#  ScoreRelevantEvents         2.5093
#  NumberOfThreadViews         0.1888
#  TimeSinceLast               0.0000
  
  
varImp(model_rf)

#======================================================================== 
#         step 2.1: Use classifier to predict progress for test data
#======================================================================== 
  
  testDb=read.csv('/home/nevena/Desktop/Digital education/DELA_Project/Classification/OutputTableTest.csv', stringsAsFactors = F)
  testDb$Grade=NULL; testDb$GradeDiff=NULL;
  testDb[is.na(testDb)]=0
  
  #---- use trained model to predict progress for test data
  preds_rf = predict(model_rf, newdata=testDb);
  
#======================================================================== 
#         step 2.2: prepare submission file for kaggle
#======================================================================== 
  
  cl.Results=testDb[,c('ProblemID', 'UserID', 'SubmissionNumber')]
  cl.Results$improved=preds_rf
  levels(cl.Results$improved)=c(0,1) # 
  cl.Results$uniqRowID= paste0(cl.Results$UserID,'_', cl.Results$ProblemID,'_', cl.Results$SubmissionNumber)
  cl.Results=cl.Results[,c('uniqRowID','improved')]
  table(cl.Results$improved)
  
  #----- keep only rows which are listed in classifier_templtae.csv file
  #----- this excludes first submissions and cases with no forum and video event in between two submissions
  classifier_template= read.csv('/home/nevena/Desktop/Digital education/DELA_Project/Classification/classifier_template.csv', stringsAsFactors = F)
  kaggleSubmission=merge(classifier_template,cl.Results )
  write.csv(kaggleSubmission,file='/home/nevena/Desktop/Digital education/DELA_Project/Classification/classifier_results_rf.csv', row.names = F)
  
  
  #------- submit the resulting file (classifier_results.csv) to kaggle 
  #------- report AUC in private score in your report
  
  
  