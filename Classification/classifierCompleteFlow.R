library(plyr) #ddply
library(dplyr)
library(AUC) #install.packages('AUC')

#======================================================================== 
#         step 1: train classifier
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
  
  # ----- (Optional) split your training data into train and test set. Use train set to build your classifier and try it on test data to check generalizability. 
  set.seed(1234)
  tr.index= sample(nrow(db), nrow(db)*0.9)
  db.train= db[tr.index,]
  db.test = db[-tr.index,]
  dim(db.train)
  dim(db.test)
  
  #----- train classifier to predict 'improved' status 
  #----- Try different methods, model parameters, feature sets and find the best classifier 
  #----- Use AUC as model evaluation metric
  library(caret)
  # ---- FEATURES FOR RANDOM FOREST
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
  
  paramGrid_rf <- expand.grid(.mtry = 8 ,.ntree = c(10,50))

  fs=c('TimeSinceLast','NVideoEvents','NForumEvents','NumberOfPlays','NumberOfPosts','NumberOfComments','SeenVideo','NumberOfDownloads','NumberOfPauses','NumberOfThreadViews','DurationOfVideoActivity','ScoreRelevantEvents','AverageVideoTimeDiffs')
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
  
  #### -- SVM Linear Model => Checking whether the data is linearly separable
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
#         step 2.1: Use classifier to predict progress for test data
#======================================================================== 
  
  testDb=read.csv('/media/diskD/EPFL/Fall 2016/DELA/DELA_Project/Classification/OutputTableTest.csv', stringsAsFactors = F)
  testDb$Grade=NULL; testDb$GradeDiff=NULL;
  testDb[is.na(testDb)]=0
  
  #---- use trained model to predict progress for test data
  preds_rf = predict(model_rf, newdata=testDb);
  
#======================================================================== 
#         step 2.1: prepare submission file for kaggle
#======================================================================== 
  
  cl.Results=testDb[,c('ProblemID', 'UserID', 'SubmissionNumber')]
  cl.Results$improved=preds_rf
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
  
  
  