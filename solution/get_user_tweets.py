from twitterscraper import query_tweets_from_user, query_tweets

# for tweet in query_tweets("Uro, Titan", 10):
#     print(tweet)
file = open("output.txt", "w")
tweets = query_tweets_from_user("@MTGGoldfish")
print(tweets, len(tweets))

for tweet in tweets:
    file.write(str(tweet.text.encode('utf-8')))
    print(str(tweet.text.encode('utf-8')), "nn")

file.close()