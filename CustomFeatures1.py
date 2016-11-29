#----------------------------------------#
# Function that computes custom features #
#----------------------------------------#
def CalculateFeatures(VideoEvents=[], ForumEvents=[]):

	# Initialize features dict
	Features = {}

	# Features for video events
	if len(VideoEvents)>0:

		# Calculate custom features
		# Keys: TimeStamp, EventType, VideoID, CurrentTime, OldTime, NewTime, SeekType, OldSpeed, NewSpeed
		TimeStamps = VideoEvents['TimeStamp']
		TimeStampDiffs = [x[0]-x[1] for x in zip(TimeStamps[1:],TimeStamps[:-1])]
		DurationOfVideoActivity = TimeStamps[-1] - TimeStamps[0]
		AverageVideoTimeDiffs = sum(TimeStampDiffs)/max(1,len(TimeStampDiffs))
		EventsTypes = VideoEvents['EventType']
		# [NEW] ADDED NEW FEATURE: NUMBER OF PAUSES

		NumberOfPauses = EventsTypes.count('Video.Pause')

		# [NEW] ADDED NEW FEATURE: NUMBER OF PLAYS
		NumberOfPlays = EventsTypes.count('Video.Play')

		# [NEW] ADDED NEW FEATURE:

		# Append features to dictionary
		Features.update({
			'DurationOfVideoActivity' : DurationOfVideoActivity,
			'AverageVideoTimeDiffs' : AverageVideoTimeDiffs,
			'NumberOfPauses': NumberOfPauses,
			'NumberOfPlays': NumberOfPlays
		})

	# Features for forum events
	if len(ForumEvents)>0:

		# Calculate custom features
		# Keys: TimeStamp, EventType, PostType, PostID, PostLength
		EventTypes = ForumEvents['EventType']
		NumberOfThreadViews = EventTypes.count('Forum.Thread.View')

		# [NEW] ADDED NEW FEATURE: NUMBER OF POSTS
		PostTypes = ForumEvents['PostType']
		NumberOfPosts = EventTypes.count('Post')

		# [NEW] ADDED NEW FEATURE: NUMBER OF COMMENTS
		NumberOfComments = EventTypes.count('Comment')

		# [NEW] ADDED NEW FEATURE: NUMBER OF THREADS
		NumberOfThreads = EventTypes.count('Thread')

		# [NEW] ADDED NEW FEATURE: NUMBER OF FORUMS
		NumberOfForums = EventTypes.count('Forum')

		# [NEW] ADDED NEW FEATURE: DOMINANT POST TYPE
		#EventCounts = []
		#EventCounts.append(NumberOfForums)
		#EventCounts.append(NumberOfThreads)
		#EventCounts.append(NumberOfPosts)
		#EventCounts.append(NumberOfComments)
		#maxCount = max(EventCounts)
		#for index in range(0,len(EventCounts)):
		#	if EventCounts[index] == maxCount:
		#		dominantIndex = index
        #dict = {
		#	0 : 'Forum',
		#	1 : 'Thread',
		#	2 : 'Post',
		#	3 : 'Comment'
		#}
        #dominantPostType = dict[dominantIndex]
    # [NEW] ADDED NEW FEATURE: DIFFFERENCE BETWEEN TIMESTAMPS
        #TimeStampsForum = ForumEvents['TimeStamp']
        #DurationOfForumActivity = TimeStampsForum[-1] - TimeStampsForum[0]

	# Append features to dictionary
        Features.update({
			'NumberOfThreadViews' : NumberOfThreadViews,
			#'NumberOfPosts':NumberOfPosts,
			#'NumberOfComments':NumberOfComments
			#'DurationOfForumActivity': DurationOfForumActivity
			#'DominantPostType': dominantPostType
		})

	return Features
