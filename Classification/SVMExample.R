# Example of SVM hyperparameter tuning and performance measures for a binary class problem with the caret package
# Rainhard Findling
# 2014 / 07
# 
library(caret)
library(plyr)
library(ROCR)

# optional - enable multicore processing
library(doMC)
registerDoMC(cores=4)

data(iris)
# create binary problem: only use two classes
x <- iris[,1:2][iris$Species %in% c('versicolor', 'virginica'),]
y <- factor(iris[,5][iris$Species %in% c('versicolor', 'virginica')])

# we see that data will probably not be perfectly seperable using linear separation
featurePlot(x = x, y = y)

# split into train and test data. train = cv parameter grid search and model creation, test = performance analysis
set.seed(123456)
indexes_y_test <- createDataPartition(y = 1:length(y), times = 1, p = 0.3)[[1]]

# creation of weights - also fast for very big datasets
weights <- as.numeric(y[-indexes_y_test])
for(val in unique(weights)) {weights[weights==val]=1/sum(weights==val)*length(weights)/2} # normalized to sum to length(samples)

model <- train(method = 'svmLinear', 
               x = x[-indexes_y_test,], 
               y = y[-indexes_y_test], 
               #weights = weights,
               maximize = T,
               tuneGrid = expand.grid(.C=3^(-15:15)),   
               preProcess = c('center', 'scale'),
               trControl = trainControl(method = 'cv', # cross validation
                                        number = 10,   # nr of cv sets
                                        #                                     repeats = 5, # use with method=repeatcv
                                        returnResamp = 'none', # return accuracy per cv partition and parameter setting
                                        classProbs = T, # return prediction probabilities along with predicted classes
                                        #                                     savePredictions=T, # returns all predictions (for all cv paritions) for each tuneGrid parameter set 
                                        returnData = F, # disable return of training data e.g. for big data sets
                                        allowParallel = T
               )
)
# we see some accuracy around 0.7-0.8
model
head(with(model, results[order(results$Kappa, decreasing=T),]))
# confusion matrix: model predicting classes of test data
table(predict.train(object = model, newdata = x[indexes_y_test,], type='raw'), y[indexes_y_test])
# prediction probabilities of test data classes
probs <- predict.train(object = model, newdata = x[indexes_y_test,], type='prob')[,1]
isPositiveClass <- y[indexes_y_test] == 'versicolor' # for a ROC curve there is a positive class (true match rate...) - defining that class here
pred <- prediction(probs, isPositiveClass)
perf <- performance(pred, 'tpr', 'fpr')
# plot: either
plot(perf, lwd=2, col=3)
# or
with(attributes(perf), plot(x=x.values[[1]], y=y.values[[1]], type='l')) 

# some metrics: AUC (area under curve), EER (equal error rate), MSER (minimum squared error rate), Cohen's Kappa etc.
AUC <- attributes(performance(pred, 'auc'))$y.value[[1]] # area under curve
df <- with(attributes(pred), data.frame(cutoffs=cutoffs[[1]], tp=tp[[1]], fn=fn[[1]], tn=tn[[1]], fp=fp[[1]], TMR=tp[[1]]/(tp[[1]]+fn[[1]]), TNMR=tn[[1]]/(tn[[1]]+fp[[1]])))
df$MSER <- with(df, sqrt((1-TMR)**2+(1-TNMR)**2))
MSER <- with(df, sqrt(TMR**2+TNMR**2)) # sqrt of minimum squared error rate = eucl. distance to point TMR=TNMR=1
i_eer <- with(df, which.min(abs(TMR-TNMR)))
EER <- with(df[i_eer,], mean(c(TMR,TNMR))) # equal error rate: mean would not be required when using ROCR as it's always exact
df$acc <- with(df, (tp + tn) / length(isPositiveClass)) # observed accuracy
df$acc_expected <- with(df, sum(isPositiveClass) * (tp+fp) / length(isPositiveClass) + sum(!isPositiveClass) * (tn+fn) / length(isPositiveClass)) / length(isPositiveClass) # expected accuracy
df$kappa <- with(df, (acc - acc_expected) / (1 - acc_expected)) # cohen's kappa
# graphical representation 
matplot(df[,6:11], lty=1:6, lwd=2, type='l'); legend('bottomright', legend=names(df[,6:11]), lty=1:6, lwd=2, col=1:6)

# characteristics for settings with best EER, MSER etc.
df[i_eer,] # curve point for equal error rate
df[order(df$acc, decreasing=T)[[1]],] # curve point for max accuracy
df[order(df$kappa, decreasing=T)[[1]],] # curve point for max kappa
df[order(df$MSER, decreasing=F)[[1]],] # curve point for minimum squared error rate