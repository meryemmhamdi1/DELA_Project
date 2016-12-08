library(plyr) #ddply
library(dplyr)
library(AUC) #install.packages('AUC')
library(caret)
library(nnet)

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

# ---- FEATURES FOR CONVOLUTIONAL NEURAL NETWORKS

# ------------ 
# ----- Whole Feature Set:
# 'ProblemID','UserID','SubmissionNumber','TimeStamp','TimeSinceLast'
# 'NVideoEvents','NForumEvents','NumberOfPlays','NumberOfPosts'
# 'NumberOfComments','SeenVideo','NumberOfDownloads','NumberOfPauses','NumberOfThreadViews'
# 'DurationOfVideoActivity','ScoreRelevantEvents','AverageVideoTimeDiffs'


fs=c('TimeSinceLast','NVideoEvents','NForumEvents','NumberOfPlays','NumberOfPosts','NumberOfComments','SeenVideo','NumberOfDownloads','NumberOfPauses','NumberOfThreadViews','DurationOfVideoActivity','ScoreRelevantEvents','AverageVideoTimeDiffs')

idC <-class.ind(db.train$improved)
nn_model = nnet(db.train[,fs],idC, size=10, maxit = 200, softmax=TRUE)

preds_train_nn = predict(nn_model, data = db.train, type="class");
ROC_curve_train_nn = roc(preds_train_nn, db.train$improved); auc(ROC_curve_train_nn)

#----- check generalizability of your model on new data
preds_test_nn= predict(nn_model, data=db.test, type = "class");
dim(preds_train_nn)
table(preds_test_nn)

ROC_curve_test_nn = roc(preds_test_nn, db.test$improved);  auc(ROC_curve_test_nn)

#======================================================================== 
#         step 1.2: Plot Confusion Matrix
#======================================================================== 
table(preds_train_nn,db.train$improved)
table(preds_test_nn,db.test$improved)


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


