library(dplyr)
library(plyr) #ddply
#------ read data frame
db=read.csv('OutputTable.csv')
#------ sort submissions
db=db[order(db$UserID,db$ProblemID,db$SubmissionNumber,db$improved),]
dim(db)
#View(db)
#------- aggregate by UserID and ProblemID ---------
  length(unique(db$UserID))
  agg.features=ddply(db, .(UserID,ProblemID,improved), summarise, 
          overalGradeDiff=Grade[length(Grade)]-Grade[1], 
          CountofSubmissions=length(SubmissionNumber),
          countOfVideoandForumEvents= (sum(NVideoEvents,na.rm = T)+sum(NForumEvents,na.rm = T)),
          # Average Time between resubmissions
          avgTbwSubs = (sum(TimeSinceLast,na.rm = T)/length(SubmissionNumber)))
#------ remove cases with only one attempt
  agg.features=filter(agg.features,CountofSubmissions>1); dim(agg.features)
#------ save feature file
  write.csv(agg.features, file='features.csv')
 