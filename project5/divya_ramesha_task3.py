from collections import defaultdict
import tweepy
import random


class MyStreamListener(tweepy.StreamListener):

    def __init__(self):
        super(MyStreamListener, self).__init__()
        self.sequenceNumber = 0
        self.sampleList = []
        self.fixedSize = 150

    def get_hash_tags(self, status):
        hashtags = []
        for tag in status.entities["hashtags"]:
            hashtags.append(tag["text"])
        return hashtags


    def on_status(self, status):
        hashtags = self.get_hash_tags(status)
        if len(hashtags) > 0:
            self.sequenceNumber += 1

            if len(self.sampleList) < self.fixedSize:
                self.sampleList.append(hashtags)
            else:
                randomValue = random.randint(0, self.sequenceNumber - 1)
                if randomValue < self.fixedSize:
                    self.sampleList[randomValue] = hashtags

            print("\nThe number of tweets with tags from the beginning: ", self.sequenceNumber)

            tagsDict = defaultdict(int)
            for hashtags in self.sampleList:
                for tag in hashtags:
                    tagsDict[str(tag)] += 1

            if len(tagsDict) > 0:
                tagFrequenciesSorted = sorted(set(tagsDict.values()), reverse=True)
                tagsByFrequency = defaultdict(list)
                for tag, frequency in tagsDict.items():
                    tagsByFrequency[frequency].append(tag)

                maxTagCount = min(len(tagFrequenciesSorted), 5)
                for top5Frequency in tagFrequenciesSorted[:maxTagCount]:
                    tagsSorted = sorted(tagsByFrequency[top5Frequency])
                    for tag in tagsSorted:
                        print(tag + " : " + str(top5Frequency))
        return True

    def on_error(self, status_code):
        print('Got an error with status code: ' + str(status_code))
        return True

    def on_timeout(self):
        print('Timeout...')
        return True


if __name__ == '__main__':
    listener = MyStreamListener()
    auth = tweepy.OAuthHandler("3SBqcl6afatylSAKc8tovAykY", "VOFZo2fWwpHkM2Lxt0XR9jN6KprkHj1KEyH8Ap3yrAYOpyEUSP")
    auth.set_access_token("2382457262-nc3aYQIuRqck3OZXuDRJ608sXptPkxB6XdBAEdY", "fkgfmmtp7EM9ddQdX0yko0C291Q44898v4in5ERU9ggCZ")

    stream = tweepy.Stream(auth, listener)
    stream.filter(track=["#"], languages=["en"])