from pyspark.mllib.recommendation import ALS, Rating
from pyspark.sql import SparkSession
from itertools import combinations
from operator import add
import random
import time
import math


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


def model_based_cf(trainFilePath, testFilePath, outputFile):

    start = time.time()
    outFile = open(outputFile, "w")

    spark = SparkSession.builder.appName('inf-553-3b').getOrCreate()
    sc = spark.sparkContext

    trainRdd = sc.textFile(trainFilePath)
    trainHeader = trainRdd.first()
    trainDataset = trainRdd.filter(lambda row: row != trainHeader).map(lambda x: x.split(","))

    testRdd = sc.textFile(testFilePath)
    testHeader = testRdd.first()
    testDataset = testRdd.filter(lambda row: row != testHeader).map(lambda x: x.split(","))

    trainUserIds = trainDataset.map(lambda x: x[0]).distinct().collect()
    trainBusinessIds = trainDataset.map(lambda x: x[1]).distinct().collect()

    testUserIds = testDataset.map(lambda x: x[0]).distinct().collect()
    testBusinessIds = testDataset.map(lambda x: x[1]).distinct().collect()

    unionUserIds = set(trainUserIds + testUserIds)
    unionBusinessIds = set(trainBusinessIds + testBusinessIds)

    userIdsMap = {}
    for index, userId in enumerate(unionUserIds):
        userIdsMap[userId] = index
    businessIdsMap = {}
    for index, businessId in enumerate(unionBusinessIds):
        businessIdsMap[businessId] = index

    testData = testDataset.map(lambda x: (userIdsMap[x[0]], businessIdsMap[x[1]]))
    testDatasetPredictions = testDataset.map(lambda x: ((userIdsMap[x[0]], businessIdsMap[x[1]]), float(x[2])))

    testDataTuple = testData.map(lambda x: (x, None))
    trainRatings = trainDataset.map(lambda x: Rating(userIdsMap[x[0]], businessIdsMap[x[1]], float(x[2])))

    def cutPrediction(x):
        if x[2] > 5:
            rate = 5
        elif x[2] < 0:
            rate = 0
        else:
            rate = x[2]
        return (x[0], x[1]), rate

    rank = 2
    numIterations = 20
    model = ALS.train(trainRatings, rank, numIterations)

    predictedData = model.predictAll(testData).map(cutPrediction)
    nonPredictedData = testDataTuple.subtractByKey(predictedData).map(lambda x: (x[0], 3))
    totalpredictionData = sc.union([predictedData, nonPredictedData])

    newPredictionData = totalpredictionData.map(lambda x: ((x[0][0], x[0][1]), float(x[1])))
    ratesAndPreds = testDatasetPredictions.join(newPredictionData)
    ratingsDifference = ratesAndPreds.map(lambda r: abs(r[1][0] - r[1][1]))

    MSE = ratingsDifference.map(lambda x: x ** 2).mean()
    RMSE = pow(MSE, 0.5)

    unionBusinessIds = list(unionBusinessIds)
    outFile.write("user_id, business_id, prediction")
    for data in totalpredictionData.collect():
        outFile.write("\n" + unionBusinessIds[data[0][0]] + "," + unionBusinessIds[data[0][1]] + "," + str(data[1]))
    outFile.close()

    print("RMSE: ", RMSE)
    print("Duration: ", (time.time() - start))


def getPearsonUserRating(user, business, userBusinessMapBroadCast, businessUserMapBroadcast):

    userBusinessMap = userBusinessMapBroadCast.value
    businessUserMap = businessUserMapBroadcast.value

    businessAverageRating = 0.0
    if businessUserMap.get(business):
        businessRatingsByOtherUsers = businessUserMap[business].values()
        businessAverageRating = sum(businessRatingsByOtherUsers) / len(businessRatingsByOtherUsers)

    if userBusinessMap.get(user):
        businessMap = userBusinessMap[user]
        userTotalRating = sum(businessMap.values())
        userAverageRating = userTotalRating / len(businessMap)

        if businessUserMap.get(business):

            averageOfBoth = (businessAverageRating + userAverageRating) / 2

            listOfOtherUsers = list(businessUserMap.get(business))
            if len(listOfOtherUsers) != 0:
                ratingDifference, intermediateListToPredictRating = list(), list()

                for otherUser in listOfOtherUsers:
                    otherUserRatings = userBusinessMap[otherUser].values()
                    otherUserAverageRating = sum(otherUserRatings) / len(otherUserRatings)

                    coRatedBusinesses = set(userBusinessMap[user].keys()) & set(userBusinessMap[otherUser].keys())

                    # userBusinessRatings, otherUserBusinessRatings = list(), list()
                    numerator, denominator = 0.0, 0.0
                    userDenominator, otherUserDenominator = 0.0, 0.0
                    for userBusiness in coRatedBusinesses:
                        if businessUserMap[userBusiness].get(user) and businessUserMap[userBusiness].get(otherUser):
                            userBusinessRatings = businessUserMap[userBusiness].get(user)
                            otherUserBusinessRatings = businessUserMap[userBusiness].get(otherUser)
                            numeratorLeft = userBusinessRatings - userAverageRating
                            numeratorRight = otherUserBusinessRatings - otherUserAverageRating
                            numerator += (numeratorLeft * numeratorRight)
                            userDenominator += (numeratorLeft * numeratorLeft)
                            otherUserDenominator += (numeratorRight * numeratorRight)

                    denominator = math.sqrt(userDenominator) * math.sqrt(otherUserDenominator)
                    pearsonCorrelation = 0
                    if denominator != 0:
                        pearsonCorrelation = numerator/denominator

                    ratingDifference = (userBusinessMap[otherUser].get(business) - otherUserAverageRating) * pearsonCorrelation
                    intermediateListToPredictRating.append((ratingDifference, abs(pearsonCorrelation)))

                predictRatingNumerator = sum(ratingDifference for ratingDifference, pearsonCorrelation in intermediateListToPredictRating)
                predictRatingDenominator = sum(pearsonCorrelation for ratingDifference, pearsonCorrelation in intermediateListToPredictRating)

                if len(intermediateListToPredictRating) == 0 or predictRatingNumerator == 0 or predictRatingDenominator == 0:
                    return (user, business), float(averageOfBoth)
                else:
                    predictedRating = (predictRatingNumerator / predictRatingDenominator)
                    predictedRating += userAverageRating
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
            return (user, business), float("2.5")


def used_based_cf(trainFilePath, testFilePath, outputFile):

    start = time.time()
    outFile = open(outputFile, "w")

    spark = SparkSession.builder.appName('inf-553-3b').getOrCreate()
    sc = spark.sparkContext

    trainRdd = sc.textFile(trainFilePath)
    trainHeader = trainRdd.first()
    trainDataset = trainRdd.filter(lambda row: row != trainHeader).map(lambda x: x.split(","))

    testRdd = sc.textFile(testFilePath)
    testHeader = testRdd.first()
    testDataset = testRdd.filter(lambda row: row != testHeader).map(lambda x: x.split(","))

    userBusinessDict = trainDataset.map(lambda x: (x[0], (x[1], float(x[2])))).groupByKey().mapValues(dict).collectAsMap()
    businessUserDict = trainDataset.map(lambda x: (x[1], (x[0], float(x[2])))).groupByKey().mapValues(dict).collectAsMap()

    testDatasetPredictions = testDataset.map(lambda x: ((x[0], x[1]), float(x[2])))

    broadcastUserBusinessMap, broadcastBusinessUserMap = sc.broadcast(userBusinessDict), sc.broadcast(businessUserDict)
    calculatedPredictions = testDataset.map(lambda x: getPearsonUserRating(x[0], x[1], broadcastUserBusinessMap, broadcastBusinessUserMap))

    outFile.write("user_id, business_id, prediction")
    for data in calculatedPredictions.collect():
        outFile.write("\n" + data[0][0] + "," + data[0][1] + "," + str(data[1]))
    outFile.close()

    ratingsDifference = testDatasetPredictions.join(calculatedPredictions).map(lambda x: abs(x[1][0] - x[1][1]))

    MSE = ratingsDifference.map(lambda x: x ** 2).mean()
    RMSE = pow(MSE, 0.5)
    print("RMSE: ", RMSE)
    print("Duration: ", time.time() - start)


def getPearsonItemRating(user, business, userBusinessMapBroadcast, businessUserMapBroadcast, businessNeighborsListBroadcast):

    userBusinessMap = userBusinessMapBroadcast.value
    businessUserMap = businessUserMapBroadcast.value
    businessNeighborsMap = businessNeighborsListBroadcast.value

    if userBusinessMap.get(user):

        businessMap = userBusinessMap.get(user)
        userTotalRating = sum(businessMap.values())
        businessList = businessMap.keys()
        userBusinessCount = len(businessList)
        userAverageRating = userTotalRating / userBusinessCount

        if business not in businessUserMap or businessUserMap.get(business) is None or len(businessUserMap.get(business)) == 0 or business not in businessNeighborsMap:
            return (user, business), float(userAverageRating)
        else:
            similarBusinessList = businessNeighborsMap[business]
            similarBusinessList = set(similarBusinessList) & set(businessList)
            if len(similarBusinessList) != 0:
                ratingDifference, intermediateListToPredictRating = list(), list()

                for similarBusiness in similarBusinessList:

                    coRatedUsers = set(businessUserMap[business].keys()) & set(businessUserMap[similarBusiness].keys())

                    businessRatings, similarBusinessRatings = list(), list()
                    totalBusinessRating, totalSimilarBusinessRating = 0, 0
                    for coratedUser in coRatedUsers:
                        if userBusinessMap[coratedUser].get(business) and userBusinessMap[coratedUser].get(similarBusiness):
                            totalBusinessRating += userBusinessMap[coratedUser].get(business)
                            totalSimilarBusinessRating += userBusinessMap[coratedUser].get(similarBusiness)
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

                if predictRatingNumerator == 0 or predictRatingDenominator == 0:
                    return (user, business), float(userAverageRating)
                else:
                    predictedRating = (predictRatingNumerator / predictRatingDenominator)
                    if predictedRating > 5.0:
                        predictedRating = 5.0
                    elif predictedRating < 0.0:
                        predictedRating = 0.0
                    return (user, business), float(predictedRating)
            else:
                return (user, business), float(userAverageRating)
    else:
        return (user, business), float("3.0")


def item_based_cf(trainFilePath, testFilePath, outputFile):

    start = time.time()
    outFile = open(outputFile, "w")

    spark = SparkSession.builder.appName('inf-553-3b').getOrCreate()
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

    MSE = ratingsDifference.map(lambda x: x ** 2).mean()
    RMSE = pow(MSE, 0.5)
    print("RMSE: ", RMSE)
    print("Duration: ", time.time() - start)
