library(caret)
rm(list=ls())
library(AUC)


#------ read features extracted from train set, using your python script

db <- read.csv("/media/diskD/EPFL/Fall 2016/DELA/DELA_Project/Classification/OutputTableTrain.csv", stringsAsFactors = TRUE)
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
db$improved = levels(factor(ifelse(db$GradeDiff>0 ,"Yes", "No" )))
table(db$improved)


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
     'NumberOfSpeedChange',
     'improved')

fs_important = c('SeenVideo',
                 'TotalVideoEvents',
                 'TotalForumEvents',
                 'EngagementIndex',
                 'NVideoAndForum_',
                 'AverageVideoTimeDiffs',
                 'SubmissionNumber',
                 'improved')


fs=c('SeenVideo',
     'DurationOfVideoActivity',
     'AverageVideoTimeDiffs',
     'DistinctIds',
     'NVideoAndForum_',
     'SelectiveNumOfEvents')


train <- db[,fs]

set.seed(999)

ind <- sample(2, nrow(train), replace=T, prob=c(0.60,0.40))
trainData<-train[ind==1,]
testData <- train[ind==2,]

set.seed(999)
ind1 <- sample(2, nrow(testData), replace=T, prob=c(0.50,0.50))
trainData_ens1<-testData[ind1==1,]
testData_ens1 <- testData[ind1==2,]

fitControl <- trainControl(method = "repeatedcv", number = 4, repeats = 4)

set.seed(33)

model_rf =train(x=trainData[,0:18],
                y=trainData$improved,
                method = "gbm",
                trControl = fitControl,
                preProc = c("center", "scale"),
                verbose = FALSE)

objControl <- trainControl(method='cv', number=3, returnResamp='none', summaryFunction = twoClassSummary, classProbs = TRUE)

objModel <- train(trainData[,0:18], as.factor(trainData$improved), 
                  method='gbm', 
                  trControl=objControl,  
                  metric = "ROC",
                  preProc = c("center", "scale"),
                  verbose = FALSE)

predictions1 <- predict(object=objModel, testData[,0:18], type='raw')
head(predictions1)

gbmFit1 <- train(as.factor(trainData$improved) ~ ., data = trainData, method = "gbm", trControl = fitControl,verbose = TRUE)


preds_train_rf = predict(model_rf, newdata=trainData[,0:7]);
ROC_curve_train_rf = roc(predictions, trainData$improved); auc(ROC_curve_train_rf)

gbm_dev <- predict(model_rf, trainData[,0:7],type= "prob");


ROC_curve_train_gbm = roc(factor(predictions1), factor(trainData$improved)); auc(ROC_curve_train_gbm)

auc(trainData$improved,predictions);

