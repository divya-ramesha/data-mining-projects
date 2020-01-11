from pyspark.sql import SparkSession
from itertools import combinations
from operator import add
import random
import time
import math
import sys

def pairs_by_lsh(dataset):

    numOfHashFuncs = 100
    numOfBands = 50
    numOfRowsPerBand = int(numOfHashFuncs / numOfBands)

    dataset = dataset.map(lambda x: (x[0], x[1]))

    userIds = dataset.map(lambda x: x[0]).distinct().collect()
    businessIds = dataset.map(lambda x: x[1]).distinct().collect()
    businessIds = list(businessIds)
    businessIds.sort()

    userIdsMap = {}
    for index, userId in enumerate(userIds):
        userIdsMap[userId] = index

    businessIdsMap = {}
    for index, businessId in enumerate(businessIds):
        businessIdsMap[businessId] = index

    userIdsCount = len(userIds)
    businessIdsCount = len(businessIds)

    businessUserList = dataset.map(lambda x: (businessIdsMap[x[1]], [userIdsMap[x[0]]])).reduceByKey(
        lambda x, y: x + y).sortBy(lambda x: x[0])
    businessUserMatrix = businessUserList.collect()

    hashValues = [(random.randint(1, businessIdsCount), random.randint(1, businessIdsCount)) for _ in
                  range(numOfHashFuncs)]

    def hashFunction(userList, hashValue):
        return min([(((hashValue[0] * usr) + hashValue[1]) % userIdsCount) for usr in userList])

    signatureMatrix = businessUserList.map(lambda x: (x[0], [hashFunction(x[1], h) for h in hashValues]))

    def getSignatureForBand(x):
        bandSignature = []
        for bandIndex in range(numOfBands):
            bandSignature.append(
                ((bandIndex, tuple(x[1][bandIndex * numOfRowsPerBand: (bandIndex + 1) * numOfRowsPerBand])), [x[0]]))
        return bandSignature

    def getPairs(x):
        candidatePairSet = []
        for comb in combinations(set(x[1]), 2):
            comb = tuple(sorted(comb))
            candidatePairSet.append((comb, 1))
        return candidatePairSet

    candidatePairs = signatureMatrix.flatMap(getSignatureForBand).reduceByKey(lambda x, y: x + y).filter(
        lambda x: len(x[1]) > 1).flatMap(getPairs).reduceByKey(add).map(lambda x: x[0])

    def computeJaccard(pair):
        list1, list2 = businessUserMatrix[pair[0]][1], businessUserMatrix[pair[1]][1]
        a = set(list1)
        b = set(list2)
        inter = a & b
        union = a | b
        return businessIds[pair[0]], businessIds[pair[1]], len(inter) / len(union)

    similarPairs = candidatePairs.map(lambda x: computeJaccard(x)).filter(lambda x: x[2] >= 0.5)
    return similarPairs


def getPearsonItemRating(user, business, userBusinessMapBroadcast, businessUserMapBroadcast, businessNeighborsListBroadcast):

    userBusinessMap = userBusinessMapBroadcast.value
    businessUserMap = businessUserMapBroadcast.value
    businessNeighborsMap = businessNeighborsListBroadcast.value

    businessAverageRating = 0.0
    if businessUserMap.get(business):
        businessRatingsByOtherUsers = businessUserMap[business].values()
        businessAverageRating = sum(businessRatingsByOtherUsers) / len(businessRatingsByOtherUsers)


    if userBusinessMap.get(user):

        businessMap = userBusinessMap.get(user)
        userTotalRating = sum(businessMap.values())
        userAverageRating = userTotalRating / len(businessMap)

        if businessUserMap.get(business):

            averageOfBoth = (businessAverageRating + userAverageRating) / 2
            businessList = businessMap.keys()

            if not businessNeighborsMap.get(business):
                return (user, business), float(averageOfBoth)
            else:
                similarBusinessList = businessNeighborsMap[business]
                similarBusinessList = set(similarBusinessList) & set(businessList)

            if len(similarBusinessList) != 0:
                intermediateListToPredictRating = list()

                for similarBusiness in similarBusinessList:

                    coRatedUsers = set(businessUserMap[business].keys()) & set(businessUserMap[similarBusiness].keys())

                    if len(coRatedUsers) == 0:
                        coRatedUsers = set(businessUserMap[business].keys())

                    businessRatings, similarBusinessRatings = list(), list()
                    for coratedUser in coRatedUsers:
                        if userBusinessMap[coratedUser].get(business) and userBusinessMap[coratedUser].get(similarBusiness):
                            businessRatings.append(userBusinessMap[coratedUser].get(business))
                            similarBusinessRatings.append(userBusinessMap[coratedUser].get(similarBusiness))

                    if businessRatings:
                        averageBusinessRating = sum(businessRatings) / len(businessRatings)
                        averageSimilarBusinessRating = sum(similarBusinessRatings) / len(similarBusinessRatings)

                        numerator, userDenominator, otherUserDenominator = 0, 0, 0
                        for index in range(len(businessRatings)):
                            numeratorLeft = businessRatings[index] - averageBusinessRating
                            numeratorRight = similarBusinessRatings[index] - averageSimilarBusinessRating
                            numerator += (numeratorLeft * numeratorRight)
                            userDenominator += (numeratorLeft * numeratorLeft)
                            otherUserDenominator += (numeratorRight * numeratorRight)

                        denominator = math.sqrt(userDenominator) * math.sqrt(otherUserDenominator)
                        pearsonCorrelation = 0
                        if denominator != 0:
                            pearsonCorrelation = numerator/denominator

                        ratingByUser = userBusinessMap[user][similarBusiness] * pearsonCorrelation
                        intermediateListToPredictRating.append((ratingByUser, abs(pearsonCorrelation)))

                predictRatingNumerator = sum(ratingByUser for ratingByUser, pearsonCorrelation in intermediateListToPredictRating)
                predictRatingDenominator = sum(pearsonCorrelation for ratingByUser, pearsonCorrelation in intermediateListToPredictRating)

                if len(intermediateListToPredictRating) == 0 or predictRatingNumerator == 0 or predictRatingDenominator == 0:
                    return (user, business), float(averageOfBoth)
                else:
                    predictedRating = (predictRatingNumerator / predictRatingDenominator)
                    if predictedRating > 5.0:
                        predictedRating = 5.0
                    elif predictedRating < 0.0:
                        predictedRating = 0.0
                    return (user, business), float(predictedRating)
            else:
                return (user, business), float(averageOfBoth)
        else:
            return (user, business), float(userAverageRating)
    else:
        if businessUserMap.get(business):
            return (user, business), float(businessAverageRating)
        else:
            return (user, business), float("3.0")


if __name__ == "__main__":
    start = time.time()

    trainFilePath = sys.argv[1] + "/yelp_train.csv"
    testFilePath = sys.argv[2]
    outputFile = sys.argv[3]
    outFile = open(outputFile, "w")

    spark = SparkSession.builder.appName('inf-553-project').getOrCreate()
    sc = spark.sparkContext

    trainRdd = sc.textFile(trainFilePath)
    trainHeader = trainRdd.first()
    trainDataset = trainRdd.filter(lambda row: row != trainHeader).map(lambda x: x.split(","))

    similarPairs = pairs_by_lsh(trainDataset)
    business1List = similarPairs.map(lambda x: (x[0], [x[1]]))
    business2List = similarPairs.map(lambda x: (x[1], [x[0]]))
    businessNeighborsList = business1List.union(business2List).reduceByKey(add).collectAsMap()

    testRdd = sc.textFile(testFilePath)
    testHeader = testRdd.first()
    testDataset = testRdd.filter(lambda row: row != testHeader).map(lambda x: x.split(","))

    userBusinessDict = trainDataset.map(lambda x: (x[0], (x[1], float(x[2])))).groupByKey().mapValues(dict).collectAsMap()
    businessUserDict = trainDataset.map(lambda x: (x[1], (x[0], float(x[2])))).groupByKey().mapValues(dict).collectAsMap()

    testDatasetPredictions = testDataset.map(lambda x: ((x[0], x[1]), float(x[2])))

    broadcastUserBusinessMap, broadcastBusinessUserMap, broadcastBusinessNeighborsList = sc.broadcast(userBusinessDict), sc.broadcast(businessUserDict), sc.broadcast(businessNeighborsList)
    calculatedPredictions = testDataset.map(lambda x: getPearsonItemRating(x[0], x[1], broadcastUserBusinessMap, broadcastBusinessUserMap, broadcastBusinessNeighborsList))

    outFile.write("user_id, business_id, prediction")
    for data in calculatedPredictions.collect():
        outFile.write("\n" + data[0][0] + "," + data[0][1] + "," + str(data[1]))
    outFile.close()

    ratingsDifference = testDatasetPredictions.join(calculatedPredictions).map(lambda x: abs(x[1][0] - x[1][1]))

    diff01 = ratingsDifference.filter(lambda x: 0 <= x < 1).count()
    diff12 = ratingsDifference.filter(lambda x: 1 <= x < 2).count()
    diff23 = ratingsDifference.filter(lambda x: 2 <= x < 3).count()
    diff34 = ratingsDifference.filter(lambda x: 3 <= x < 4).count()
    diff4 = ratingsDifference.filter(lambda x: 4 <= x).count()

    MSE = ratingsDifference.map(lambda x: x ** 2).mean()
    RMSE = pow(MSE, 0.5)

    print("\n>=0 and <1:", diff01)
    print(">=1 and <2:", diff12)
    print(">=2 and <3:", diff23)
    print(">=3 and <4:", diff34)
    print(">=4:", diff4)
    print("RMSE: ", RMSE)
    print("Execution Time: ", time.time() - start)
