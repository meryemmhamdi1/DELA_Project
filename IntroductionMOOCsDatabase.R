# 1. Add Additional Features (Optional)

# 2. Train and evaluate a classifier to predict if grade improves after a resubmission

# 2.1. Train with linear model using caret package

# Partition the data into train(70%) and test(30% sets) using createDataPartition()
set.seed(134)
tr.index = createDataPartition(y=agg.features$Price, p=0.70, list = F)
d.train= dn[tr.index,]
d.test = dn[-tr.index,]
m2 <- train(overalGradeDiff ~ CountofSubmissions,data=agg.features,method="lm")