library(plyr) #ddply
library(dplyr)
library(AUC) #install.packages('AUC')
library(caret)
library(class)

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
  #db = db[db$SubmissionNumber>0,]
  db= filter(db,SubmissionNumber>0)
  
  #---- remove cases when there is no video or forum activity between two submissions
  db$NVideoAndForum= db$NVideoEvents+db$NForumEvents
  # db = db[db$NVideoAndForum>0,]
  db= filter(db,NVideoAndForum>0)  
  
  # "ffb9b5233042d1c0cf2277ca55241e82"
  i = 0
  for (user in unique(db$UserID)){
    i = i + 1
    db$userCode[db$UserID==user] = i 
  }
  db$userCode <- as.numeric(db$userCode)
    
  
  fs=c("ProblemID",
         "userCode",
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
  
  scalar1 <- function(x) {(x-min(x))/(max(x)-min(x))}
  for (feature in fs){
    db[feature] = scalar1(db[feature])
  }
  
  #----- make a categorical variable, indicating if grade improved
  db$improved = factor(ifelse(db$GradeDiff>0 ,"1", "0" ))
  table(db$improved)
  set.seed(9560)
  up_train <- upSample(x=db,
                       y= db$improved)
  up_train$improved = up_train$Class
  # ----- (Optional) split your training data into train and test set. Use train set to build your classifier and try it on test data to check generalizability. 
  set.seed(1234)
  tr.index= sample(nrow(up_train), nrow(up_train)*0.7)
  db.train= up_train[tr.index,]
  db.test = up_train[-tr.index,]
  dim(db.train)
  dim(db.test)
  
  # DIMENSIONALITY REDUCTION
  
  # log transform 
  log.db <- log(db[,fs])
  db.classes <- db$improved
  
  # apply PCA - scale. = TRUE is highly 
  # advisable, but default is FALSE. 
  db.pca <- prcomp(db[,fs],
                   center = TRUE,
                   scale. = TRUE) 
  
  summary(db.pca)

  plot(db.pca)
  # plot method
  par(mar = rep(2, 4))
  biplot(db.pca)
  
  
  db_pca = predict(db.pca, 
          newdata=tail(db[,fs], 2))
  
  require(caret)
  trans = preProcess(db[,fs], 
                     method=c("BoxCox", "center", 
                              "scale", "pca"))
  PC = predict(trans, db[,fs])
  
  
  library(devtools)
  install_github("ggbiplot", "vqv")
  
  library(ggbiplot)
  g <- ggbiplot(db.pca, obs.scale = 1, var.scale = 1, 
                groups = db.classes, ellipse = TRUE, 
                circle = TRUE)
  g <- g + scale_color_discrete(name = '')
  g <- g + theme(legend.direction = 'horizontal', 
                 legend.position = 'top')
  print(g)
  
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

  # removed features: 'NForumEvents', 'ScoreRelevantEvents', 'NumberOfThreadViews', 'TimeSinceLast'
  
  fs=c('SubmissionNumber',
       "TotalVideoEvents",
       "TotalForumEvents",
       'SeenVideo',
       'DurationOfVideoActivity',
       "EngagementIndex"
       )
  
  fs_4=c("ProblemID",
       "userCode",
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
  
  
  fs_3 = c("NVideoEvents",                
      "TotalVideoEvents",             
      "NVideoAndForum_",             
      "SelectiveNumOfEvents",         
      "AverageVideoTimeDiffs",        
      "DistinctIds",                  
      "DurationOfVideoActivity",      
      "SeenVideo",                    
      "PlaysDownlsPerVideo",          
      "NumberOfLoads",                
      "NumberOfPlays",                
      "NumberOfPauses",               
      "NumberOfThreadsLaunched",      
      "NumberOfSpeedChange",          
      "EngagementIndex",              
      "NumberOfDownloads",            
      "NumberOfComments",             
      "ComAndPost",                   
      "TotalForumEvents",             
      "NumberOfPosts")
  
  fs_2=c('NVideoEvents',
       'NVideoAndForum_',
       'AverageVideoTimeDiffs',
       'DistinctIds',
       'DurationOfVideoActivity',
       'SelectiveNumOfEvents')
  # TRYING KNN
  
  fs = c('NVideoEvents','NVideoAndForum_','SelectiveNumOfEvents','AverageVideoTimeDiffs','DistinctIds')
  knn_pred <- knn(train = db.train[,fs], test = db.test[,fs], cl = db.train$improved, k=15);
  ROC_curve_train_knn = roc(knn_pred, db.test$improved); auc(ROC_curve_train_knn)

  ctrl_rf= trainControl(method = 'cv', summaryFunction=twoClassSummary ,classProbs = TRUE)
  # -- Random Forest Model
  paramGrid_rf = expand.grid(.mtry = c(1:6) ,.ntree = c(10,10))
  model_rf = train(x=db.train[,fs],
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
  preds_test_rf= predict(model_rf, newdata = db.test);
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
  
  testDb=read.csv('/media/diskD/EPFL/Fall 2016/DELA/DELA_Project/Classification/OutputTableTest.csv', stringsAsFactors = F)
  testDb$ProblemIDOld = testDb$ProblemID
  testDb$SubmissionNumberOld = testDb$SubmissionNumber
  testDb$Grade=NULL; testDb$GradeDiff=NULL;
  testDb$NVideoAndForum= testDb$NVideoEvents+testDb$NForumEvents
  testDb[is.na(testDb)]=0
  
  scalar1 <- function(x) {(x-min(x))/(max(x)-min(x))}
  for (feature in fs){
    testDb[feature] = scalar1(testDb[feature])
  }
  
  #---- use trained model to predict progress for test data
  preds_rf = predict(model_rf, newdata =testDb[,fs]);
  knn_pred <- knn(train = db.train[,fs], test = testDb[,fs], cl = db.train$improved, k=10)
  

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
  classifier_template= read.csv('/media/diskD/EPFL/Fall 2016/DELA/DELA_Project/Classification/classifier_template.csv', stringsAsFactors = F)
  kaggleSubmission=merge(classifier_template,cl.Results )
  write.csv(kaggleSubmission,file='/media/diskD/EPFL/Fall 2016/DELA/DELA_Project/Classification/classifier_results_knn.csv', row.names = F)
  
  
  #------- submit the resulting file (classifier_results.csv) to kaggle 
  #------- report AUC in private score in your report
  
  
  fs=c('SubmissionNumber',
       'NVideoEvents',
       'NumberOfPosts',
       'NumberOfComments',
       'SeenVideo',
       'NumberOfDownloads',
       'NumberOfPauses',
       'DurationOfVideoActivity',
       'AverageVideoTimeDiffs',
       'DistinctIds',
       'PlaysDownlsPerVideo',
       'ComAndPost',
       'NumberOfThreadsLaunched',
       'NumberOfLoads',
       'NumberOfPlays',
       'NVideoAndForum_',
       'SelectiveNumOfEvents',
       'NumberOfSpeedChange')
  
  set.seed(7)
  # load the library
  library(mlbench)
  library(caret)
  # define the control using a random forest selection function
  control <- rfeControl(functions=rfFuncs, method="cv", number=10)
  # run the RFE algorithm
  results <- rfe(db.train[,fs], db.train$improved, sizes=c(1:18), rfeControl=control)
  # summarize the results
  print(results)
  # list the chosen features
  predictors(results)
  
  # plot the results
  plot(results, type=c("g", "o"))
  # plot the results
  plot(results, type=c("g", "o"))
  
  