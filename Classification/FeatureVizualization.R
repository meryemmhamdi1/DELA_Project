fs=c('TimeSinceLast','NVideoEvents','NForumEvents','NumberOfPlays','NumberOfPosts','NumberOfComments','SeenVideo','NumberOfDownloads','NumberOfPauses','NumberOfThreadViews','DurationOfVideoActivity','ScoreRelevantEvents','AverageVideoTimeDiffs')

# Normalization of features:
scalar1 <- function(x) {(x-min(x))/(max(x)-min(x))}
for (feature in fs){
  db[feature] = scalar1(db[feature])
}


dbYes = db[db$improved=='Yes',]

dbNo = db[db$improved=='No',]

#---  AverageVideoTimeDiffs     100.0000
#---  DurationOfVideoActivity    98.8803
#---  NVideoEvents               98.1915
#---  SeenVideo                  95.0247
#---  NumberOfPlays              87.1058
#---  NumberOfPauses             78.0625
#---  NumberOfComments           69.2735
#---  NumberOfDownloads          69.1281
#---  NumberOfPosts              63.7541

summary(dbYes$AverageVideoTimeDiffs)
summary(dbNo$AverageVideoTimeDiffs)

summary(dbYes$DurationOfVideoActivity)
summary(dbNo$DurationOfVideoActivity)

summary(dbYes$NVideoEvents)
summary(dbNo$NVideoEvents)

summary(dbYes$SeenVideo)
summary(dbNo$SeenVideo)

summary(dbYes$NumberOfPlays)
summary(dbNo$NumberOfPlays)

summary(dbYes$NumberOfPauses)
summary(dbNo$NumberOfPauses)

summary(dbYes$NumberOfComments)
summary(dbNo$NumberOfComments)

summary(dbYes$NumberOfDownloads)
summary(dbNo$NumberOfDownloads)

summary(dbYes$NumberOfPosts)
summary(dbNo$NumberOfPosts)


length(unique(dbYes$AverageVideoTimeDiffs))
length(unique(dbNo$AverageVideoTimeDiffs))

length(unique(dbYes$DurationOfVideoActivity))
length(unique(dbNo$DurationOfVideoActivity))

length(unique(dbYes$NVideoEvents))
length(unique(dbNo$NVideoEvents))

length(unique(dbYes$SeenVideo))
length(unique(dbNo$SeenVideo))

length(unique(dbYes$NumberOfPlays))
length(unique(dbNo$NumberOfPlays))

length(unique(dbYes$NumberOfPauses))
length(unique(dbNo$NumberOfPauses))

length(unique(dbYes$NumberOfComments))
length(unique(dbNo$NumberOfComments))

length(unique(dbYes$NumberOfDownloads))
length(unique(dbNo$NumberOfDownloads))

length(unique(dbYes$NumberOfPosts))
length(unique(dbNo$NumberOfPosts))

#======================================================================== 
#         step 1: Plot 2 Dimensional Features
#======================================================================== 

plot(db$SeenVideo,type = "b")
text(db$SeenVideo,labels = db$improved,col = c('red','green')[y])