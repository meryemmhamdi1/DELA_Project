# ----------------------------------------#
# Function that computes custom features #
# ----------------------------------------#
def CalculateFeatures(VideoEvents=[], ForumEvents=[],NVideoAndForum_=0):
    # Initialize features dict
    Features = {}
   # NVideoAndForum_=len(VideoEvents)+len(ForumEvents)
    # Features for video events
    if len(VideoEvents) > 0:

        # Calculate custom features
        # Keys: TimeStamp, EventType, VideoID, CurrentTime, OldTime, NewTime, SeekType, OldSpeed, NewSpeed
        TimeStamps = VideoEvents['TimeStamp']
        TimeStampDiffs = [x[0] - x[1] for x in zip(TimeStamps[1:], TimeStamps[:-1])]
        DurationOfVideoActivity = TimeStamps[-1] - TimeStamps[0]
        AverageVideoTimeDiffs = sum(TimeStampDiffs) / max(1, len(TimeStampDiffs))

        EventsTypes = VideoEvents['EventType']
        
        # [NEW] ADDED NEW FEATURE: NUMBER OF PAUSES
        NumberOfPauses = EventsTypes.count('Video.Pause')

        # [NEW] ADDED NEW FEATURE: NUMBER OF PLAYS
        NumberOfPlays = EventsTypes.count('Video.Play')

        # [NEW] ADDED NEW FEATURE: NUMBER OF DOWNLOADS
        NumberOfDownloads = EventsTypes.count('Video.Download')

        # [NEW] ADDED NEW FEATURE: NUMBER OF LOADS
        NumberOfLoads= EventsTypes.count('Video.Load')

        # [NEW] ADDED NEW FEATURE: NUMBER OF SPEED CHANGES
        NumberOfSpeedChange= EventsTypes.count('Video.SpeedChange')

        # [NEW] ADDED NEW FEATURE: WATCHED (SEEN OR DOWNLOADED) AT LEAST ONE VIDEO
        SeenVideo = 0
        if NumberOfPlays > 0 or NumberOfDownloads > 0:
            SeenVideo = 1

        # [NEW] ADDED NEW FEATURE: NUMBER OF DISTINCT VIDEO IDS
        DistinctIds=len(set(VideoEvents['VideoID']))
        
        # [NEW] ADDED NEW FEATURE: PLAYS AND DOWNLOADS PER VIDEO
        PlaysDownlsPerVideo=(NumberOfPlays+NumberOfDownloads)/DistinctIds

        # [NEW] ADDED NEW FEATURE: PLAYS AND DOWNLOADS PER VIDEO
        SelectiveNumOfEvents=NumberOfPlays+NumberOfPauses+NumberOfLoads+NumberOfSpeedChange

        # Append features to dictionary
        Features.update({
            'DurationOfVideoActivity': DurationOfVideoActivity,
            'AverageVideoTimeDiffs': AverageVideoTimeDiffs,
            'NumberOfPlays': NumberOfPlays,
            'NumberOfDownloads': NumberOfDownloads,
            'NumberOfPauses': NumberOfPauses,
            'SeenVideo': SeenVideo,
            'DistinctIds': DistinctIds,
            'PlaysDownlsPerVideo':PlaysDownlsPerVideo,
            'NumberOfLoads':NumberOfLoads,
            'NumberOfSpeedChange': NumberOfSpeedChange,
            'SelectiveNumOfEvents': SelectiveNumOfEvents,
            'NVideoAndForum_':NVideoAndForum_
        })

    # Features for forum events
    if len(ForumEvents) > 0:
        # Calculate custom features
        # Keys: TimeStamp, EventType, PostType, PostID, PostLength
        EventTypes = ForumEvents['EventType']
        NumberOfThreadViews = EventTypes.count('Forum.Thread.View')

        PostTypes = ForumEvents['PostType']
        # [NEW] ADDED NEW FEATURE: COUNT NUMBER OF COMMENTS
        NumberOfComments = PostTypes.count('Comment')

        # [NEW] ADDED NEW FEATURE: COUNT NUMBER OF POSTS
        NumberOfPosts = PostTypes.count('Post')

        # [NEW] ADDED NEW FEATURE: WEIGHTED SUM OF RELEVANT POST-TYPES
        ScoreRelevantEvents = 2 * NumberOfComments + 1.5 * NumberOfPosts + 1 * NumberOfThreadViews

        #[NEW] ADDED NEW FEATURE: COUNT ONLY POSTS AND COMMENTS
        ComAndPost=NumberOfComments+NumberOfPosts
        
        #[NEW] ADDED NEW FEATURE: COUNT ONLY POSTS AND COMMENTS
        NumberOfThreadsLaunched = EventTypes.count('Forum.Thread.Launch')
        
        # Append features to dictionary
        Features.update({
            'NumberOfThreadViews': NumberOfThreadViews,
            'NumberOfComments': NumberOfComments,
            'NumberOfPosts': NumberOfPosts,
            'ScoreRelevantEvents': ScoreRelevantEvents,
            'ComAndPost':ComAndPost,
            'NumberOfThreadsLaunched':NumberOfThreadsLaunched
        })

    return Features
